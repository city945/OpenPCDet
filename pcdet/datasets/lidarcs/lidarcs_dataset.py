import numpy as np
import copy

from pcdet.datasets import DatasetTemplate
from pcdet.utils import misc_utils

class LidarCSDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.load_infos(self.mode)
        if dataset_cfg.get('MIX_SENSOR', None) and training:
            import pandas as pd
            import os
            path_prefix = f"../{dataset_cfg.MIX_SENSOR.SENSOR}"
            mix_sensor_root = self.root_path / path_prefix
            mix_infos = pd.read_pickle(mix_sensor_root / 'lidarcs_infos_train.pkl')
            num_mix_frames, mix_type = dataset_cfg.MIX_SENSOR.NUM_FRAMES, dataset_cfg.MIX_SENSOR.MIX_TYPE
            for i in range(num_mix_frames):
                mix_infos[i]['lidar']['filepath'] = os.path.join(path_prefix, mix_infos[i]['lidar']['filepath'])
            if mix_type == 'replace':
                self.infos[:num_mix_frames] = mix_infos[:num_mix_frames]
            elif mix_type == 'append':
                self.infos.extend(mix_infos)

        self.mix_gt = False
        if dataset_cfg.get('MIX_GT', None) and dataset_cfg.MIX_GT.SENSOR != self.root_path.name:
            import pandas as pd
            import os
            assert (dataset_cfg.MIX_GT.MIX_TYPE == 'replace')
            path_prefix = f"../{dataset_cfg.MIX_GT.SENSOR}"
            mix_sensor_root = self.root_path / path_prefix
            info_path = self.dataset_cfg.INFO_PATH[self.mode][0]
            mix_infos = pd.read_pickle(mix_sensor_root / info_path)
            self.raw_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]
            for i in range(len(mix_infos)):
                mix_infos[i]['lidar']['filepath'] = os.path.join(path_prefix, mix_infos[i]['lidar']['filepath'])
                self.infos[i]['annos'] = mix_infos[i]['annos']
            self.mix_infos = mix_infos
            self.mix_gt = True

    @staticmethod
    def get_lidar(filepath, num_features=4):
        return np.fromfile(filepath, dtype=np.float32).reshape(-1, num_features)

    @staticmethod
    def get_label(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        name_list, gt_boxes_lidar_list = [], []
        for line in lines:
            label = line.strip().split(' ')
            gt_boxes_lidar = np.array([float(label[1]), float(label[2]), float(label[3]), float(label[4]), float(label[5]), float(label[6]), float(label[7])])
            name_list.append(label[0])
            gt_boxes_lidar_list.append(gt_boxes_lidar)
            
        return {'name': np.array(name_list), 'gt_boxes_lidar': np.array(gt_boxes_lidar_list)}

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):

        info = copy.deepcopy(self.infos[index])

        input_dict = {
            'frame_id': info['lidar']['frame_id'],
        }

        points = self.get_lidar(self.root_path / info['lidar']['filepath'])
        if self.mix_gt:
            mix_points = self.get_lidar(self.root_path / self.mix_infos[index]['lidar']['filepath'])
            mix_gt_boxes_lidar = self.mix_infos[index]['annos']['gt_boxes_lidar']
            raw_gt_boxes_lidar = self.raw_gt_annos[index]['gt_boxes_lidar']
            points = misc_utils.replace_gt(points, raw_gt_boxes_lidar, mix_points, mix_gt_boxes_lidar)

        input_dict['points'] = points

        if 'annos' in info:
            annos = info['annos']

            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (annos['num_points_in_gt'] >= self.dataset_cfg.FILTER_MIN_POINTS_IN_GT)
                annos['name'] = annos['name'][mask]
                annos['gt_boxes_lidar'] = annos['gt_boxes_lidar'][mask]

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['gt_boxes_lidar'],
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]
        if kwargs['eval_metric'] == 'lidarcs':
            from .lidarcs_eval import get_evaluation_results
            ap_result_str, ap_dict = get_evaluation_results(eval_gt_annos, eval_det_annos, class_names)
            return ap_result_str, ap_dict
        else:
            raise NotImplementedError