import torch
import glob, os

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu

def save_checkpoint(model, optimizer, scaler, lr_scheduler, lr_warmup_scheduler, eval_state, epoch, it=None, filename='checkpoint'):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    state = {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}
    state['scaler_state'] = scaler.state_dict()
    state['scheduler_state'] = lr_scheduler.state_dict()
    state['warmup_scheduler_state'] = lr_warmup_scheduler.state_dict() if lr_warmup_scheduler is not None else None
    state['eval_state'] = eval_state
    torch.save(state, f"{filename}.pth")

def load_checkpoint(ckpt_dir, model, optimizer, scaler, lr_scheduler, lr_warmup_scheduler, to_cpu=True):
    def resume(filename):
        start_it, start_epoch = model.load_params_with_optimizer(filename, to_cpu=to_cpu, optimizer=optimizer)
        checkpoint = torch.load(filename, map_location=(torch.device('cpu') if to_cpu else None))
        scaler.load_state_dict(checkpoint['scaler_state'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state'])
        if checkpoint['warmup_scheduler_state'] is not None:
            lr_warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state'])
        return start_it, start_epoch

    start_epoch, start_it, eval_states, resumed = 0, 0, [], False
    ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
    if len(ckpt_list) > 0:
        ckpt_list.sort(key=os.path.getmtime)
        while len(ckpt_list) > 0:
            try:
                start_it, start_epoch = resume(ckpt_list[-1])
                resumed = True
                break
            except:
                ckpt_list = ckpt_list[:-1]
        for filename in ckpt_list:
            checkpoint = torch.load(filename, map_location=(torch.device('cpu') if to_cpu else None))
            if checkpoint['eval_state'] is not None:
                eval_states.append(checkpoint['eval_state'])
    return start_epoch, start_it, eval_states, resumed

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

