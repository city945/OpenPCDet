import numpy as np
import copy

from pcdet.datasets import DatasetTemplate

class LidarCSDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.load_infos(self.mode)

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