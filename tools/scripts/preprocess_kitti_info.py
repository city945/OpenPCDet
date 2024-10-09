import os, pickle
from pcdet.datasets import KittiDataset as dataset
from pcdet.utils import box_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import numpy as np
from pathlib import Path

kitti_classes = ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 'Person_sitting', 'Tram', 'Misc']

def get_infos(sample_id_list, data_root_path, num_workers=4, has_label=True, count_inside_pts=True):

    def process_single_scene(sample_idx):
        print('sample_idx: %s' % sample_idx)
        info = {}
        pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
        info['point_cloud'] = pc_info

        image_info = {'image_idx': sample_idx, 'image_shape': dataset.get_image_shape(data_root_path, sample_idx)}
        info['image'] = image_info
        calib = dataset.get_calib(data_root_path, sample_idx)

        P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
        R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
        R0_4x4[3, 3] = 1.
        R0_4x4[:3, :3] = calib.R0
        V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
        calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

        info['calib'] = calib_info

        if has_label:
            obj_list = dataset.get_label(data_root_path, sample_idx)
            annotations = {}
            annotations['name'] = np.array([obj.cls_type for obj in obj_list])
            annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
            annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
            annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
            annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
            annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
            annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
            annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
            annotations['score'] = np.array([obj.score for obj in obj_list])
            annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

            num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
            num_gt = len(annotations['name'])
            index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
            annotations['index'] = np.array(index, dtype=np.int32)

            loc = annotations['location'][:num_objects]
            dims = annotations['dimensions'][:num_objects]
            rots = annotations['rotation_y'][:num_objects]
            loc_lidar = calib.rect_to_lidar(loc)
            l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
            loc_lidar[:, 2] += h[:, 0] / 2
            gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
            annotations['gt_boxes_lidar'] = gt_boxes_lidar

            info['annos'] = annotations

            if count_inside_pts:
                points = dataset.get_lidar(data_root_path, sample_idx)
                calib = dataset.get_calib(data_root_path, sample_idx)
                pts_rect = calib.lidar_to_rect(points[:, 0:3])

                fov_flag = dataset.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                pts_fov = points[fov_flag]
                corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                for k in range(num_objects):
                    flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                    num_points_in_gt[k] = flag.sum()
                annotations['num_points_in_gt'] = num_points_in_gt

        return info

    import concurrent.futures as futures
    with futures.ThreadPoolExecutor(num_workers) as executor:
        infos = executor.map(process_single_scene, sample_id_list)
    return list(infos)

def create_groundtruth_database(data_root_path, save_path, info_path=None, used_classes=None, split='train'):
    import torch

    database_save_path = Path(save_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
    db_info_save_path = Path(save_path) / ('kitti_dbinfos_%s.pkl' % split)

    database_save_path.mkdir(parents=True, exist_ok=True)
    all_db_infos = {}

    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    for k in range(len(infos)):
        print('gt_database sample: %d/%d' % (k + 1, len(infos)))
        info = infos[k]
        sample_idx = info['point_cloud']['lidar_idx']
        points = dataset.get_lidar(data_root_path, sample_idx)
        annos = info['annos']
        names = annos['name']
        difficulty = annos['difficulty']
        bbox = annos['bbox']
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
                            'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
    for k, v in all_db_infos.items():
        print('Database %s: %d' % (k, len(v)))

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)

def create_kitti_infos(data_path, save_path, workers=8):
    split_dict = {
        'train': 'ImageSets/train.txt', 
        'val': 'ImageSets/val.txt', 
        'test': 'ImageSets/test.txt'
        }
    for split, split_file in split_dict.items():
        save_filename = save_path / f"kitti_infos_{split}.pkl"
        
        sample_id_list = [x.strip() for x in open(save_path / split_file).readlines()]
        data_root_path = data_path / ('training' if split != 'test' else 'testing')

        infos = get_infos(sample_id_list, data_root_path,
            num_workers=workers, has_label=(split != 'test'), count_inside_pts=True)
        with open(save_filename, 'wb') as f:
            pickle.dump(infos, f)
        print(f'Kitti info {split} file is saved to {save_filename}')

        if split == 'train':
            print('Start create groundtruth database for data augmentation')
            create_groundtruth_database(data_root_path, save_path, info_path=save_filename, split=split)

def create_kitti_vis_infos(save_path, valid_classes=kitti_classes, num_features=4):
    """
    transform KITTI infos format for visualizer
    """
    for split in ['train', 'val']:
        input_file = save_path / f"kitti_infos_{split}.pkl"
        output_file = save_path / f"kitti_vis_infos_{split}.pkl"
        infos = pickle.load(open(input_file, 'rb'))

        split_key = "training" if (split in ['train', 'val']) else "testing"
        vis_infos = []
        for info in infos:
            frame_id = info['point_cloud']['lidar_idx']
            cloudpath = f"{split_key}/velodyne/{frame_id}.bin"
            lidar_info = {
                'frame_id': frame_id,
                'filepath': cloudpath,
                'num_features': num_features,
            }

            imagepath = f"{split_key}/image_2/{info['image']['image_idx']}.png"
            lidar2pixel_mat = info['calib']['P2'] @ info['calib']['R0_rect'] @ info['calib']['Tr_velo_to_cam']
            image_info = {
                'image_2': {
                    'filepath': imagepath,
                    'l2p_mat': lidar2pixel_mat,
                },
            }

            if 'annos' in info:
                annos = {}
                gt_boxes_lidar = []
                for i, name in enumerate(info['annos']['name']):
                    if name in valid_classes:
                        gt_boxes_lidar.append(info['annos']['gt_boxes_lidar'][i])
                annos['gt_boxes_lidar'] = np.array(gt_boxes_lidar)

                mask = np.array([name in valid_classes for name in info['annos']['name']], dtype=bool)
                for k, v in info['annos'].items():
                    if k in ['name', 'num_points_in_gt', 'difficulty', 'occluded', 'truncated']:
                        annos[k] = v[mask]
            vis_infos.append({'lidar': lidar_info, 'image': image_info, 'annos': annos})
        
        with open(output_file, 'wb') as f:
            pickle.dump(vis_infos, f)
        print(f"create {output_file}")

if __name__ == '__main__':
    import argparse
    from pcdet.config import cfg

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--func', type=str, default='create_kitti_infos', help='or create_kitti_vis_infos')

    args = parser.parse_args()

    if args.func == 'create_kitti_infos':
        create_kitti_infos(
            data_path=cfg.ROOT_DIR / 'data' / 'kitti',
            save_path=cfg.ROOT_DIR / 'data' / 'kitti',
        )
    elif args.func == 'create_kitti_vis_infos':
        create_kitti_vis_infos(
            save_path=cfg.ROOT_DIR / 'data' / 'kitti',
            valid_classes=kitti_classes,
        )
    else:
        raise NotImplementedError
