import os, pickle
from pcdet.datasets import LidarCSDataset as dataset
from pcdet.utils import box_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import numpy as np
from pathlib import Path

lidarcs_classes = ['Car', 'Pedestrian', 'Bicyclist', 'Motorcyclist', 'Truck', 'TrafficCone']

def get_infos(sample_id_list, data_root_path, num_workers=4, has_label=True, count_inside_pts=True):

    def process_single_scene(sample_idx):
        print('sample_idx: %s' % sample_idx)
        info = {}
        lidar_info = {
            'num_features': 4, 
            'frame_id': sample_idx,
            'filepath': os.path.join('bin', f'{sample_idx}.bin'),
            }
        info['lidar'] = lidar_info

        if has_label:
            annos = dataset.get_label(data_root_path / 'label' / f'{sample_idx}.txt')

            if count_inside_pts:
                points = dataset.get_lidar(data_root_path / lidar_info['filepath'])
                num_objects = len(annos['name'])

                corners_lidar = box_utils.boxes_to_corners_3d(annos['gt_boxes_lidar'])
                num_points_in_gt = -np.ones(num_objects, dtype=np.int32)

                for k in range(num_objects):
                    flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                    num_points_in_gt[k] = flag.sum()
                annos['num_points_in_gt'] = num_points_in_gt

            info['annos'] = annos

        return info

    import concurrent.futures as futures
    with futures.ThreadPoolExecutor(num_workers) as executor:
        infos = executor.map(process_single_scene, sample_id_list)
    return list(infos)

def create_groundtruth_database(data_root_path, save_path, info_path=None, used_classes=None, split='train'):
    import torch

    database_save_path = Path(save_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
    db_info_save_path = Path(save_path) / ('lidarcs_dbinfos_%s.pkl' % split)

    database_save_path.mkdir(parents=True, exist_ok=True)
    all_db_infos = {}

    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    for k in range(len(infos)):
        print('gt_database sample: %d/%d' % (k + 1, len(infos)))
        info = infos[k]
        sample_idx = info['lidar']['frame_id']
        points = dataset.get_lidar(data_root_path / info['lidar']['filepath'])
        annos = info['annos']
        names = annos['name']
        gt_boxes = annos['gt_boxes_lidar']

        num_obj = gt_boxes.shape[0]
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
        ).numpy()  # (nboxes, npoints)

        for i in range(num_obj):
            filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
            filepath = database_save_path / filename
            gt_points = points[point_indices[i] > 0]

            gt_points[:, :3] -= gt_boxes[i, :3]
            with open(filepath, 'w') as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_path = str(filepath.relative_to(save_path))  # gt_database/xxxxx.bin
                db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                            }
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
    for k, v in all_db_infos.items():
        print('Database %s: %d' % (k, len(v)))

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)

def create_lidarcs_infos(data_path, save_path, sensor, workers=8):
    # split_dict = {
    #     'train': 'splits/train.txt', 
    #     'val': 'splits/test.txt', 
    #     }
    split_dict = {
        'train': 'ImageSets/train.txt', 
        'val': 'ImageSets/val.txt', 
        'test': 'ImageSets/test.txt', 
        }
    for split, split_file in split_dict.items():
        save_filename = save_path / f"lidarcs_infos_{split}.pkl"
        
        sample_id_list = [x.strip() for x in open(data_path / split_file).readlines()]
        data_root_path = data_path / sensor

        infos = get_infos(sample_id_list, data_root_path,
            num_workers=workers, has_label=True, count_inside_pts=True)
        with open(save_filename, 'wb') as f:
            pickle.dump(infos, f)
        print(f'Lidar-CS info {split} file is saved to {save_filename}')

        if split == 'train':
            print('Start create groundtruth database for data augmentation')
            create_groundtruth_database(data_root_path, save_path, info_path=save_filename, split=split)


if __name__ == '__main__':
    import argparse
    from pcdet.config import cfg

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--func', type=str, default='create_lidarcs_infos', help='')
    parser.add_argument('--sensor', type=str, default='VLD-64', help='[VLD-16, VLD-32, VLD-64, VLD-128]')

    args = parser.parse_args()

    if args.func == 'create_lidarcs_infos':
        create_lidarcs_infos(
            data_path=cfg.ROOT_DIR / 'data' / 'lidarcs',
            save_path=cfg.ROOT_DIR / 'data' / 'lidarcs' / args.sensor,
            sensor=args.sensor,
        )
    else:
        raise NotImplementedError
