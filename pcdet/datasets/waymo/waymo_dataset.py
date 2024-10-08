# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved.

import os
import pickle
import copy
import numpy as np
import torch
import multiprocessing
import SharedArray
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from functools import partial

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.infos = []
        self.seq_name_to_infos = self.include_waymo_data(self.mode)

        self.use_shared_memory = self.dataset_cfg.get('USE_SHARED_MEMORY', False) and self.training
        if self.use_shared_memory:
            self.shared_memory_file_limit = self.dataset_cfg.get('SHARED_MEMORY_FILE_LIMIT', 0x7FFFFFFF)
            self.load_data_to_shared_memory()

        if self.dataset_cfg.get('USE_PREDBOX', False):
            self.pred_boxes_dict = self.load_pred_boxes_to_dict(
                pred_boxes_path=self.dataset_cfg.ROI_BOXES_PATH[self.mode]
            )
        else:
            self.pred_boxes_dict = {}
        self.disable_nlz = dataset_cfg.get('DISABLE_NLZ_FLAG_ON_POINTS', False)
        self.points_tanh_dim = self.dataset_cfg.get('POINTS_TANH_DIM', None)

    def include_waymo_data(self, mode):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []
        seq_name_to_infos = {}

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = self.data_path / sequence_name / ('%s.pkl' % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

            seq_name_to_infos[infos[0]['point_cloud']['lidar_sequence']] = infos

        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))
            
        if self.dataset_cfg.get("DEBUG", False):
            self.infos = self.infos[:16]
            
        use_sequence_data = self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED
        if not use_sequence_data:
            seq_name_to_infos = None 
        return seq_name_to_infos

    def load_pred_boxes_to_dict(self, pred_boxes_path):
        self.logger.info(f'Loading and reorganizing pred_boxes to dict from path: {pred_boxes_path}')
        with open(pred_boxes_path, 'rb') as f:
            pred_dicts = pickle.load(f)

        pred_boxes_dict = {}
        for index, box_dict in enumerate(pred_dicts):
            seq_name = box_dict['frame_id'][:-4].replace('training_', '').replace('validation_', '')
            sample_idx = int(box_dict['frame_id'][-3:])

            if seq_name not in pred_boxes_dict:
                pred_boxes_dict[seq_name] = {}

            pred_labels = np.array([self.class_names.index(box_dict['name'][k]) + 1 for k in range(box_dict['name'].shape[0])])
            pred_boxes = np.concatenate((box_dict['boxes_lidar'], box_dict['score'][:, np.newaxis], pred_labels[:, np.newaxis]), axis=-1)
            pred_boxes_dict[seq_name][sample_idx] = pred_boxes

        self.logger.info(f'Predicted boxes has been loaded, total sequences: {len(pred_boxes_dict)}')
        return pred_boxes_dict

    def load_data_to_shared_memory(self):
        self.logger.info(f'Loading training data to shared memory (file limit={self.shared_memory_file_limit})')

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = self.infos[:self.shared_memory_file_limit] \
            if self.shared_memory_file_limit < len(self.infos) else self.infos
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            sa_key = f'{sequence_name}___{sample_idx}'
            if os.path.exists(f"/dev/shm/{sa_key}"):
                continue

            points = self.get_lidar(self.data_path, sequence_name, sample_idx, self.disable_nlz, self.points_tanh_dim)
            common_utils.sa_create(f"shm://{sa_key}", points)

        dist.barrier()
        self.logger.info('Training data has been saved to shared memory')

    def clean_shared_memory(self):
        self.logger.info(f'Clean training data from shared memory (file limit={self.shared_memory_file_limit})')

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = self.infos[:self.shared_memory_file_limit] \
            if self.shared_memory_file_limit < len(self.infos) else self.infos
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            sa_key = f'{sequence_name}___{sample_idx}'
            if not os.path.exists(f"/dev/shm/{sa_key}"):
                continue

            SharedArray.delete(f"shm://{sa_key}")

        if num_gpus > 1:
            dist.barrier()
        self.logger.info('Training data has been deleted from shared memory')

    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        # 兼容各个版本数据集文件的命名与存放位置，如果提供的文件无法找到则查找其他版本的模式下能否找到该文件
        if not sequence_file.exists():
            found_sequence_file = sequence_file
            for pre_text in ['training', 'validation', 'testing']:
                temp_sequence_file = sequence_file.parent / pre_text / sequence_file.name
                if temp_sequence_file.exists():
                    found_sequence_file = temp_sequence_file
                    break
                if not sequence_file.exists():
                    temp_sequence_file = Path(str(sequence_file).replace('segment', pre_text + '_segment'))
                    if temp_sequence_file.exists():
                        found_sequence_file = temp_sequence_file
                        break
            if not found_sequence_file.exists():
                found_sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))
            if found_sequence_file.exists():
                sequence_file = found_sequence_file
        return sequence_file

    @staticmethod
    def get_lidar(data_path, sequence_name, sample_idx, disable_nlz=True, points_tanh_dim=None):
        lidar_file = data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        if not disable_nlz:
            points_all = points_all[NLZ_flag == -1]
        if points_tanh_dim is None:
            points_all[:, 3] = np.tanh(points_all[:, 3])
        else:
            for dim_idx in points_tanh_dim:
                points_all[:, dim_idx] = np.tanh(points_all[:, dim_idx])
        return points_all

    @staticmethod
    def transform_prebox_to_current(pred_boxes3d, pose_pre, pose_cur):
        """

        Args:
            pred_boxes3d (N, 9 or 11): [x, y, z, dx, dy, dz, raw, <vx, vy,> score, label]
            pose_pre (4, 4):
            pose_cur (4, 4):
        Returns:

        """
        assert pred_boxes3d.shape[-1] in [9, 11]
        pred_boxes3d = pred_boxes3d.copy()
        expand_bboxes = np.concatenate([pred_boxes3d[:, :3], np.ones((pred_boxes3d.shape[0], 1))], axis=-1)

        bboxes_global = np.dot(expand_bboxes, pose_pre.T)[:, :3]
        expand_bboxes_global = np.concatenate([bboxes_global[:, :3],np.ones((bboxes_global.shape[0], 1))], axis=-1)
        bboxes_pre2cur = np.dot(expand_bboxes_global, np.linalg.inv(pose_cur.T))[:, :3]
        pred_boxes3d[:, 0:3] = bboxes_pre2cur

        if pred_boxes3d.shape[-1] == 11:
            expand_vels = np.concatenate([pred_boxes3d[:, 7:9], np.zeros((pred_boxes3d.shape[0], 1))], axis=-1)
            vels_global = np.dot(expand_vels, pose_pre[:3, :3].T)
            vels_pre2cur = np.dot(vels_global, np.linalg.inv(pose_cur[:3, :3].T))[:,:2]
            pred_boxes3d[:, 7:9] = vels_pre2cur

        pred_boxes3d[:, 6]  = pred_boxes3d[..., 6] + np.arctan2(pose_pre[..., 1, 0], pose_pre[..., 0, 0])
        pred_boxes3d[:, 6]  = pred_boxes3d[..., 6] - np.arctan2(pose_cur[..., 1, 0], pose_cur[..., 0, 0])
        return pred_boxes3d

    @staticmethod
    def reorder_rois_for_refining(pred_bboxes):
        num_max_rois = max([len(bbox) for bbox in pred_bboxes])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        ordered_bboxes = np.zeros([len(pred_bboxes), num_max_rois, pred_bboxes[0].shape[-1]], dtype=np.float32)

        for bs_idx in range(ordered_bboxes.shape[0]):
            ordered_bboxes[bs_idx, :len(pred_bboxes[bs_idx])] = pred_bboxes[bs_idx]
        return ordered_bboxes

    def get_sequence_data(self, info, points, sequence_name, sample_idx, sequence_cfg, load_pred_boxes=False):
        """
        Args:
            info:
            points:
            sequence_name:
            sample_idx:
            sequence_cfg:
        Returns:
        """

        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        def load_pred_boxes_from_dict(sequence_name, sample_idx):
            """
            boxes: (N, 11)  [x, y, z, dx, dy, dn, raw, vx, vy, score, label]
            """
            sequence_name = sequence_name.replace('training_', '').replace('validation_', '')
            load_boxes = self.pred_boxes_dict[sequence_name][sample_idx]
            assert load_boxes.shape[-1] == 11
            load_boxes[:, 7:9] = -0.1 * load_boxes[:, 7:9]  # transfer speed to negtive motion from t to t-1
            return load_boxes

        pose_cur = info['pose'].reshape((4, 4))
        num_pts_cur = points.shape[0]
        sample_idx_pre_list = np.clip(sample_idx + np.arange(sequence_cfg.SAMPLE_OFFSET[0], sequence_cfg.SAMPLE_OFFSET[1]), 0, 0x7FFFFFFF)
        sample_idx_pre_list = sample_idx_pre_list[::-1]

        if sequence_cfg.get('ONEHOT_TIMESTAMP', False):
            onehot_cur = np.zeros((points.shape[0], len(sample_idx_pre_list) + 1)).astype(points.dtype)
            onehot_cur[:, 0] = 1
            points = np.hstack([points, onehot_cur])
        else:
            points = np.hstack([points, np.zeros((points.shape[0], 1)).astype(points.dtype)])
        points_pre_all = []
        num_points_pre = []

        pose_all = [pose_cur]
        pred_boxes_all = []
        if load_pred_boxes:
            pred_boxes = load_pred_boxes_from_dict(sequence_name, sample_idx)
            pred_boxes_all.append(pred_boxes)

        sequence_info = self.seq_name_to_infos[sequence_name]

        for idx, sample_idx_pre in enumerate(sample_idx_pre_list):

            points_pre = self.get_lidar(self.data_path, sequence_name, sample_idx_pre, self.disable_nlz, self.points_tanh_dim)
            pose_pre = sequence_info[sample_idx_pre]['pose'].reshape((4, 4))
            expand_points_pre = np.concatenate([points_pre[:, :3], np.ones((points_pre.shape[0], 1))], axis=-1)
            points_pre_global = np.dot(expand_points_pre, pose_pre.T)[:, :3]
            expand_points_pre_global = np.concatenate([points_pre_global, np.ones((points_pre_global.shape[0], 1))], axis=-1)
            points_pre2cur = np.dot(expand_points_pre_global, np.linalg.inv(pose_cur.T))[:, :3]
            points_pre = np.concatenate([points_pre2cur, points_pre[:, 3:]], axis=-1)
            if sequence_cfg.get('ONEHOT_TIMESTAMP', False):
                onehot_vector = np.zeros((points_pre.shape[0], len(sample_idx_pre_list) + 1))
                onehot_vector[:, idx + 1] = 1
                points_pre = np.hstack([points_pre, onehot_vector])
            else:
                # add timestamp
                points_pre = np.hstack([points_pre, 0.1 * (sample_idx - sample_idx_pre) * np.ones((points_pre.shape[0], 1)).astype(points_pre.dtype)])  # one frame 0.1s
            points_pre = remove_ego_points(points_pre, 1.0)
            points_pre_all.append(points_pre)
            num_points_pre.append(points_pre.shape[0])
            pose_all.append(pose_pre)

            if load_pred_boxes:
                pose_pre = sequence_info[sample_idx_pre]['pose'].reshape((4, 4))
                pred_boxes = load_pred_boxes_from_dict(sequence_name, sample_idx_pre)
                pred_boxes = self.transform_prebox_to_current(pred_boxes, pose_pre, pose_cur)
                pred_boxes_all.append(pred_boxes)

        points = np.concatenate([points] + points_pre_all, axis=0).astype(np.float32)
        num_points_all = np.array([num_pts_cur] + num_points_pre).astype(np.int32)
        poses = np.concatenate(pose_all, axis=0).astype(np.float32)

        if load_pred_boxes:
            temp_pred_boxes = self.reorder_rois_for_refining(pred_boxes_all)
            pred_boxes = temp_pred_boxes[:, :, 0:9]
            pred_scores = temp_pred_boxes[:, :, 9]
            pred_labels = temp_pred_boxes[:, :, 10]
        else:
            pred_boxes = pred_scores = pred_labels = None

        return points, num_points_all, sample_idx_pre_list, poses, pred_boxes, pred_scores, pred_labels

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        input_dict = {
            'sample_idx': sample_idx
        }
        if self.use_shared_memory and index < self.shared_memory_file_limit:
            sa_key = f'{sequence_name}___{sample_idx}'
            points = SharedArray.attach(f"shm://{sa_key}").copy()
        else:
            points = self.get_lidar(self.data_path, sequence_name, sample_idx, self.disable_nlz, self.points_tanh_dim)

        if self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED:
            points, num_points_all, sample_idx_pre_list, poses, pred_boxes, pred_scores, pred_labels = self.get_sequence_data(
                info, points, sequence_name, sample_idx, self.dataset_cfg.SEQUENCE_CONFIG,
                load_pred_boxes=self.dataset_cfg.get('USE_PREDBOX', False)
            )
            input_dict['poses'] = poses
            if self.dataset_cfg.get('USE_PREDBOX', False):
                input_dict.update({
                    'roi_boxes': pred_boxes,
                    'roi_scores': pred_scores,
                    'roi_labels': pred_labels,
                })

        input_dict.update({
            'points': points,
            'frame_id': info['frame_id'],
        })

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            if self.dataset_cfg.get('TRAIN_WITH_SPEED', False):
                assert gt_boxes_lidar.shape[-1] == 9
            else:
                gt_boxes_lidar = gt_boxes_lidar[:, 0:7]

            if self.training and self.dataset_cfg.get('FILTER_EMPTY_BOXES_FOR_TRAIN', False):
                mask = (annos['num_points_in_gt'] > 0)  # filter empty boxes
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])
        data_dict.pop('num_points_in_gt', None)
        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict
