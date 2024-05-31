import argparse
import datetime
import glob
import os
from pathlib import Path

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator, load_data_to_gpu
from pcdet.utils import common_utils, train_utils, commu_utils
from pcdet.optim import build_optimizer, build_scheduler
import wandb
import tqdm
import time
from torch.nn.utils import clip_grad_norm_
import pickle

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--work_dir', type=str, default="output", help='the dir to save logs and models')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')
    parser.add_argument('--debug', action='store_true', default=False, help='debug setting')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='')
    parser.add_argument('--wandb_run_id', type=str, default=None, help='wandb run id for resume')

    parser.add_argument('--eval', action='store_true', default=False, help='only perform evaluate')
    parser.add_argument('--eval_skip_epochs', type=int, default=0, help='eval skip first N epochs')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)
    args.extra_tag = datetime.datetime.now().strftime('%Y%m%d-%H%M') if args.extra_tag == 'default' else args.extra_tag
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    if args.debug:
        args.batch_size = 1 if args.batch_size is None else args.batch_size
        args.workers = 0
        args.epochs = 1 if args.epochs is None else args.epochs
        args.extra_tag = 'debug'
        cfg.DATA_CONFIG.DEBUG = True

    return args, cfg

class Trainer:
    def __init__(self, args, cfg):
        self.args, self.cfg = args, cfg
        # init dist && cfg && dir && log && tb && wandb
        self._init(args, cfg)

        # dataloader
        train_set, train_loader, train_sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=self.dist_train, workers=args.workers,
            logger=self.logger,
            training=True,
            total_epochs=args.epochs,
            seed=666 if args.fix_random_seed else None
        )
        test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=self.dist_train, workers=args.workers, 
            logger=self.logger, 
            training=False,
        )

        # model && optimizer && scheduler
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()
        if args.pretrained_model is not None:
            model.load_params_from_file(filename=args.pretrained_model, to_cpu=self.dist_train, logger=self.logger)

        optimizer = build_optimizer(model, cfg.OPTIMIZATION)
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp, init_scale=cfg.OPTIMIZATION.get('LOSS_SCALE_FP16', 2.0**16))
        lr_scheduler, lr_warmup_scheduler = build_scheduler(
            optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
            optim_cfg=cfg.OPTIMIZATION
        )
        start_epoch, start_it, eval_states, resumed = \
            train_utils.load_checkpoint(self.ckpt_dir, to_cpu=self.dist_train,
                                        model=model, optimizer=optimizer, scaler=scaler,
                                        lr_scheduler=lr_scheduler, lr_warmup_scheduler=lr_warmup_scheduler,
                                        )
        model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
        if self.dist_train:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])

        # log
        if not resumed:
            self.logger.info('**********************Start logging**********************')
            gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
            self.logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
            if self.dist_train:
                self.logger.info('Training in distributed mode : total_batch_size: %d' % (self.total_gpus * args.batch_size))
            else:
                self.logger.info('Training with a single process')
            for key, val in vars(args).items():
                self.logger.info('{:16} {}'.format(key, val))
            log_config_to_file(cfg, logger=self.logger)
            self.logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
            self.logger.info(model)

        self.model_func = model_fn_decorator()
        self.train_loader, self.train_sampler, self.test_loader = train_loader, train_sampler, test_loader
        self.model, self.optimizer, self.lr_scheduler, self.lr_warmup_scheduler, self.scaler, self.eval_states = model, optimizer, lr_scheduler, lr_warmup_scheduler, scaler, eval_states
        self.start_epoch, self.it, self.total_epoch = start_epoch, start_it, args.epochs
        self.eval_skip_epochs = args.eval_skip_epochs
        self.cfg_file, self.use_amp, self.use_wandb = args.cfg_file, args.use_amp, args.use_wandb
        self.rank, self.optim_cfg = cfg.LOCAL_RANK, cfg.OPTIMIZATION

    def _init(self, args, cfg):
        # dist && cfg
        if args.launcher == 'none':
            dist_train = False
            total_gpus = 1
        else:
            total_gpus, cfg.LOCAL_RANK = common_utils.init_dist_pytorch_v1_10()
            dist_train = True

        if args.batch_size is None:
            args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
        else:
            assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
            args.batch_size = args.batch_size // total_gpus
        args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

        if args.fix_random_seed:
            common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

        # dir && log
        output_dir = cfg.ROOT_DIR / args.work_dir / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
        ckpt_dir = output_dir / 'ckpt'
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        log_file = output_dir / ('train_%s.log' % args.extra_tag)
        logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
        if cfg.LOCAL_RANK == 0:
            os.system('cp %s %s' % (args.cfg_file, output_dir))

        # tb && wandb
        tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None
        if cfg.LOCAL_RANK == 0 and args.use_wandb:
            project_name = os.path.abspath(__file__).split(os.sep)[-3]
            if args.wandb_run_id is None:
                wandb.init(project=project_name, config=cfg, dir=output_dir)
            else:
                wandb.init(project=project_name, config=cfg, dir=output_dir, resume=True, id=args.wandb_run_id)
            wandb.config.update(args, allow_val_change=True)
            wandb.run.tags = cfg.EXP_GROUP_PATH.split('/') + [cfg.TAG]
            wandb.run.name = args.extra_tag

        self.dist_train, self.total_gpus = dist_train, total_gpus
        self.output_dir, self.ckpt_dir, self.log_file, self.logger = output_dir, ckpt_dir, log_file, logger
        self.tb_log = tb_log

    def _train_one_epoch(self, cur_epoch, tbar):
        self.model.train()
        if self.rank == 0:
            pbar = tqdm.tqdm(total=len(self.train_loader), leave=(cur_epoch+1 == self.total_epoch), desc='train', dynamic_ncols=True)
            data_time = common_utils.AverageMeter()
            batch_time = common_utils.AverageMeter()
            forward_time = common_utils.AverageMeter()
            losses_m = common_utils.AverageMeter()

        total_it_each_epoch = len(self.train_loader)
        dataloader_iter = iter(self.train_loader)
        for cur_it in range(total_it_each_epoch):
            end = time.time()
            batch = next(dataloader_iter)
            data_timer = time.time()
            cur_data_time = data_timer - end
            
            self.lr_scheduler.step(self.it, cur_epoch)
            try:
                cur_lr = float(self.optimizer.lr)
            except:
                cur_lr = self.optimizer.param_groups[0]['lr']

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss, tb_dict, disp_dict = self.model_func(self.model, batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.optim_cfg.GRAD_NORM_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.it += 1
    
            cur_forward_time = time.time() - data_timer
            cur_batch_time = time.time() - end

            # average reduce
            avg_data_time = commu_utils.average_reduce_value(cur_data_time)
            avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
            avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

            if self.rank == 0:
                batch_size = batch.get('batch_size', None)
                
                data_time.update(avg_data_time)
                forward_time.update(avg_forward_time)
                batch_time.update(avg_batch_time)
                losses_m.update(loss.item() , batch_size)
                
                disp_dict.update({
                    'loss': loss.item(), 'lr': cur_lr, 
                    'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                    'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 
                    'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})',
                })
                
                pbar.update()
                pbar.set_postfix(dict(total_it=self.it))
                tbar.set_postfix(disp_dict)

                if self.tb_log is not None:
                    self.tb_log.add_scalar('train/loss', loss, self.it)
                    self.tb_log.add_scalar('meta_data/learning_rate', cur_lr, self.it)
                    for key, val in tb_dict.items():
                        self.tb_log.add_scalar('train/' + key, val, self.it)
                if self.rank == 0 and self.use_wandb:
                    wandb.log({'meta_data/learning_rate': cur_lr}, step=self.it)
                    train_metrics_by_iter = {'train/loss': loss}
                    for key, val in tb_dict.items():
                        train_metrics_by_iter.update({'train/' + key: val})
                    wandb.log(train_metrics_by_iter, step=self.it)
                    
        if self.rank == 0:
            pbar.close()

    def train(self):
        with tqdm.trange(self.start_epoch, self.total_epoch, desc='epochs', dynamic_ncols=True, leave=(self.rank == 0)) as tbar:
            for cur_epoch in tbar:
                # train one epoch
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(cur_epoch)
                if self.lr_warmup_scheduler is not None and cur_epoch < self.optim_cfg.WARMUP_EPOCH:
                    self.lr_scheduler = self.lr_warmup_scheduler
                self._train_one_epoch(cur_epoch, tbar)

                # val
                if (cur_epoch+1) > self.eval_skip_epochs:
                    ret_dict = self.evaluate(cur_epoch, self.test_loader)
                    if ret_dict is not None:
                        eval_state = {'epoch': cur_epoch+1, 'ret_dict': ret_dict}
                        self.eval_states.append(eval_state)
                    if self.dist_train:
                        torch.distributed.barrier()
                    time.sleep(1)
                else:
                    eval_state = None

                # save trained model
                if self.rank == 0:
                    ckpt_name = self.ckpt_dir / ('checkpoint_epoch_%d' % (cur_epoch+1))
                    train_utils.save_checkpoint(self.model, self.optimizer, self.scaler, self.lr_scheduler, self.lr_warmup_scheduler, eval_state, (cur_epoch+1), it=self.it, filename=ckpt_name)

        # only keep best and last model
        if self.rank == 0 and (self.cfg.get('EVAL_CONFIG', None) is not None) and (self.cfg['EVAL_CONFIG'].get('BEST_METRIC_KEYS', None) is not None):
            ckpt_infos_list = self.eval_states
            last_ckpt_epoch = ckpt_infos_list[-1]['epoch']
            best_metric_keys = self.cfg['EVAL_CONFIG']['BEST_METRIC_KEYS']
            ckpt_infos_list = sorted(ckpt_infos_list, key=lambda x: tuple(x['ret_dict'][key] for key in best_metric_keys))
            best_ckpt_epoch = ckpt_infos_list[-1]['epoch']
            os.system(f"cd {self.ckpt_dir} && cp checkpoint_epoch_{best_ckpt_epoch}.pth best_epoch_{best_ckpt_epoch}.pth")
            os.system(f"cd {self.ckpt_dir} && cp checkpoint_epoch_{last_ckpt_epoch}.pth last_epoch_{last_ckpt_epoch}.pth")
            list(map(os.remove, glob.glob(str(self.ckpt_dir / f"checkpoint_epoch_*.pth"))))

            self.logger.info(f"Best model is epoch {best_ckpt_epoch}")
            for key in best_metric_keys:
                val = ckpt_infos_list[-1]['ret_dict'][key]
                self.logger.info(f"{key}: {val:.2f}")

        if self.rank == 0 and self.use_wandb:
            file_artifact = wandb.Artifact(type='file', name=wandb.run.name)
            file_artifact.add_file(self.log_file)
            file_artifact.add_file(self.cfg_file)
            wandb.log_artifact(file_artifact)
            wandb.finish()
 
    def evaluate(self, cur_epoch, dataloader):
        self.model.eval()
        result_dir = self.output_dir / 'eval' / f'epoch_{cur_epoch+1}'
        result_dir.mkdir(parents=True, exist_ok=True)

        metric = {
            'gt_num': 0,
        }
        for cur_thresh in self.cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
            metric['recall_roi_%s' % str(cur_thresh)] = 0
            metric['recall_rcnn_%s' % str(cur_thresh)] = 0

        self.logger.info(f"*************** TRAINED EPOCH {cur_epoch+1} EVALUATION *****************")
        if self.rank == 0:
            progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)

        det_annos = []
        dataset = dataloader.dataset
        start_time = time.time()
        for i, batch_dict in enumerate(dataloader):
            load_data_to_gpu(batch_dict)

            with torch.no_grad():
                pred_dicts, ret_dict = self.model(batch_dict)

            disp_dict = {}

            train_utils.statistics_info(self.cfg, ret_dict, metric, disp_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, dataset.class_names,
                output_path=None
            )
            det_annos += annos
            if self.rank == 0:
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()

        if self.rank == 0:
            progress_bar.close()

        if self.dist_train:
            rank, world_size = common_utils.get_dist_info()
            det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
            metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

        self.logger.info(f'*************** Performance of EPOCH {cur_epoch+1} *****************')
        sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
        self.logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

        if self.rank != 0:
            return None

        ret_dict = {}
        if self.dist_train:
            for key, val in metric[0].items():
                for k in range(1, world_size):
                    metric[0][key] += metric[k][key]
            metric = metric[0]

        gt_num_cnt = metric['gt_num']
        for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
            cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            self.logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
            self.logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
            ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
            ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

        total_pred_objects = 0
        for anno in det_annos:
            total_pred_objects += anno['name'].__len__()
        self.logger.info('Average predicted number of objects(%d samples): %.3f'
                    % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

        with open(result_dir / 'result.pkl', 'wb') as f:
            pickle.dump(det_annos, f)

        result_str, result_dict = dataset.evaluation(
            det_annos, dataset.class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=None
        )

        self.logger.info(result_str)
        ret_dict.update(result_dict)

        if self.tb_log is not None:
            for key, val in ret_dict.items():
                self.tb_log.add_scalar(key, val, cur_epoch+1)
        if self.rank == 0 and self.use_wandb:
            val_metrics_by_epoch = {'eval/epoch': cur_epoch+1}
            for key, val in ret_dict.items():
                val_metrics_by_epoch.update({'eval/' + key.replace('/', '_'): val})
            
            wandb.log(val_metrics_by_epoch)

        self.logger.info('Result is saved to %s' % result_dir)
        self.logger.info('****************Evaluation done.*****************')

        return ret_dict

def main():
    args, cfg = parse_config()
    trainer = Trainer(args, cfg)

    if args.eval:
        trainer.evaluate(trainer.start_epoch-1, trainer.test_loader)
        if trainer.dist_train:
            torch.distributed.barrier()
        time.sleep(1)
    else:
        trainer.train()

if __name__ == '__main__':
    main()