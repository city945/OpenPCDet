import os, pickle
from pcdet.datasets import ONCEDataset as dataset
from pcdet.utils import box_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import numpy as np
from pathlib import Path

once_classes = ['Car', 'Bus', 'Truck', 'Pedestrian', 'Cyclist']

def get_infos(toolkits, sample_seq_list, data_root_path, num_workers=4, count_inside_pts=True):
    import json
    cam_names = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']

    def process_single_sequence(seq_idx):
        print('seq_idx: %s' % seq_idx)
        seq_infos = []
        seq_path = data_root_path / 'data' / seq_idx
        json_path = seq_path / ('%s.json' % seq_idx)
        with open(json_path, 'r') as f:
            info_this_seq = json.load(f)
        meta_info = info_this_seq['meta_info']
        calib = info_this_seq['calib']
        for f_idx, frame in enumerate(info_this_seq['frames']):
            frame_id = frame['frame_id']
            if f_idx == 0:
                prev_id = None
            else:
                prev_id = info_this_seq['frames'][f_idx-1]['frame_id']
            if f_idx == len(info_this_seq['frames'])-1:
                next_id = None
            else:
                next_id = info_this_seq['frames'][f_idx+1]['frame_id']
            pc_path = str(seq_path / 'lidar_roof' / ('%s.bin' % frame_id))
            pose = np.array(frame['pose'])
            frame_dict = {
                'sequence_id': seq_idx,
                'frame_id': frame_id,
                'timestamp': int(frame_id),
                'prev_id': prev_id,
                'next_id': next_id,
                'meta_info': meta_info,
                'lidar': pc_path,
                'pose': pose
            }
            calib_dict = {}
            for cam_name in cam_names:
                cam_path = str(seq_path / cam_name / ('%s.jpg' % frame_id))
                frame_dict.update({cam_name: cam_path})
                calib_dict[cam_name] = {}
                calib_dict[cam_name]['cam_to_velo'] = np.array(calib[cam_name]['cam_to_velo'])
                calib_dict[cam_name]['cam_intrinsic'] = np.array(calib[cam_name]['cam_intrinsic'])
                calib_dict[cam_name]['distortion'] = np.array(calib[cam_name]['distortion'])
            frame_dict.update({'calib': calib_dict})

            if 'annos' in frame:
                annos = frame['annos']
                boxes_3d = np.array(annos['boxes_3d'])
                if boxes_3d.shape[0] == 0:
                    print(frame_id)
                    continue
                boxes_2d_dict = {}
                for cam_name in cam_names:
                    boxes_2d_dict[cam_name] = np.array(annos['boxes_2d'][cam_name])
                annos_dict = {
                    'name': np.array(annos['names']),
                    'gt_boxes_lidar': boxes_3d,
                    'boxes_2d': boxes_2d_dict
                }

                if count_inside_pts:
                    points = dataset.get_lidar(toolkits, seq_idx, frame_id)
                    num_gt = boxes_3d.shape[0]

                    corners_lidar = box_utils.boxes_to_corners_3d(np.array(annos['boxes_3d']))
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_gt):
                        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annos_dict['num_points_in_gt'] = num_points_in_gt

                frame_dict.update({'annos': annos_dict})

            seq_infos.append(frame_dict)
        return seq_infos

    import concurrent.futures as futures
    with futures.ThreadPoolExecutor(num_workers) as executor:
        infos = executor.map(process_single_sequence, sample_seq_list)
    all_infos = []
    for seq_info in infos:
        all_infos.extend(seq_info)
    return all_infos

def create_groundtruth_database(toolkits, save_path, info_path=None, used_classes=None, split='train'):
    import torch

    database_save_path = Path(save_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
    db_info_save_path = Path(save_path) / ('once_dbinfos_%s.pkl' % split)

    database_save_path.mkdir(parents=True, exist_ok=True)
    all_db_infos = {}

    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    for k in range(len(infos)):
        if 'annos' not in infos[k]:
            continue
        print('gt_database sample: %d/%d' % (k + 1, len(infos)))
        info = infos[k]
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        points = dataset.get_lidar(toolkits, seq_id, frame_id)
        annos = info['annos']
        names = annos['name']
        gt_boxes = annos['gt_boxes_lidar']

        num_obj = gt_boxes.shape[0]
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
        ).numpy()  # (nboxes, npoints)

        for i in range(num_obj):
            filename = '%s_%s_%d.bin' % (frame_id, names[i], i)
            filepath = database_save_path / filename
            gt_points = points[point_indices[i] > 0]

            gt_points[:, :3] -= gt_boxes[i, :3]
            with open(filepath, 'w') as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_path = str(filepath.relative_to(save_path))  # gt_database/xxxxx.bin
                db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
    for k, v in all_db_infos.items():
        print('Database %s: %d' % (k, len(v)))

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)

def create_once_infos(data_path, save_path, workers=8):
    from pcdet.datasets.once.once_toolkits import Octopus
    split_dict = {
        'train': 'ImageSets/train.txt', 
        'val': 'ImageSets/val.txt', 
        'test': 'ImageSets/test.txt'
        }
    toolkits = Octopus(data_path)
    for split, split_file in split_dict.items():
        save_filename = save_path / f"once_infos_{split}.pkl"
        
        sample_seq_list = [x.strip() for x in open(save_path / split_file).readlines()]

        infos = get_infos(toolkits, sample_seq_list, data_root_path=data_path,
            num_workers=workers, count_inside_pts=True)
        with open(save_filename, 'wb') as f:
            pickle.dump(infos, f)
        print(f'ONCE info {split} file is saved to {save_filename}')

        if split == 'train':
            print('Start create groundtruth database for data augmentation')
            create_groundtruth_database(toolkits, save_path, info_path=save_filename, split=split)

def create_once_vis_infos(data_path, save_path, valid_classes=once_classes, num_features=4, add_ext_info=True):
    """
    transform ONCE infos format for visualizer
    """
    for split in ['train', 'val']:
        input_file = save_path / f"once_infos_{split}.pkl"
        output_file = save_path / f"once_vis_infos_{split}.pkl"
        infos = pickle.load(open(input_file, 'rb'))

        vis_infos = []
        for info in infos:
            if 'annos' not in info:
                continue
            sequence_id = info['sequence_id']
            frame_id = info['frame_id']
            cloudpath = str(Path(info['lidar']).relative_to(data_path))
            lidar_info = {
                'frame_id': f"{sequence_id}-{frame_id}",
                'filepath': cloudpath,
                'num_features': num_features,
            }

            cam_names = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
            image_info = {}
            for key in cam_names:
                imagepath = str(Path(info[key]).relative_to(data_path))
                calib_info = info['calib'][key]
                cam_2_velo = calib_info['cam_to_velo']
                cam_intri = np.zeros([4, 4], dtype=np.float32)
                cam_intri[3, 3] = 1.
                cam_intri[:3, :3] = calib_info['cam_intrinsic']
                lidar2pixel_mat = cam_intri @ np.linalg.inv(cam_2_velo)

                image_info[key] = {
                    'filepath': imagepath,
                    'l2p_mat': lidar2pixel_mat,
                }

            if 'annos' in info:
                annos = {}
                gt_boxes_lidar = []
                for i, name in enumerate(info['annos']['name']):
                    if name in valid_classes:
                        gt_boxes_lidar.append(info['annos']['boxes_3d'][i])
                annos['gt_boxes_lidar'] = np.array(gt_boxes_lidar)

                mask = np.array([name in valid_classes for name in info['annos']['name']], dtype=bool)
                for k, v in info['annos'].items():
                    if k in ['name', 'num_points_in_gt']:
                        annos[k] = v[mask]
            vis_infos.append({'lidar': lidar_info, 'image': image_info, 'annos': annos})
        
        if add_ext_info:
            for vis_info, info in zip(vis_infos, infos):
                vis_info['lidar']['desc'] = f"{info['meta_info']['weather']}, {info['meta_info']['period']}".lower()

        with open(output_file, 'wb') as f:
            pickle.dump(vis_infos, f)
        print(f"create {output_file}")

if __name__ == '__main__':
    import argparse
    from pcdet.config import cfg

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--func', type=str, default='create_once_infos', help='or create_once_vis_infos')

    args = parser.parse_args()

    if args.func == 'create_once_infos':
        create_once_infos(
            data_path=Path("/datasets/ONCE"),
            save_path=cfg.ROOT_DIR / 'data' / 'once',
        )
    elif args.func == 'create_once_vis_infos':
        create_once_vis_infos(
            data_path=Path("/datasets/ONCE"),
            save_path=cfg.ROOT_DIR / 'data' / 'once',
            valid_classes=once_classes,
        )
    else:
        raise NotImplementedError
