import os, pickle
from pcdet.datasets import WaymoDataset as dataset
from pcdet.utils import box_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import numpy as np
from pathlib import Path
import multiprocessing, tqdm

waymo_classes = ['Vehicle', 'Pedestrian', 'Cyclist', 'Sign']

def process_single_sequence(sequence_file, data_save_path, sampled_interval, has_label, update_info_only, use_two_returns=True):
    from pcdet.datasets.waymo.waymo_utils import tf, dataset_pb2, generate_labels, save_lidar_points
    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]

    # print('Load record (sampled_interval=%d): %s' % (sampled_interval, sequence_name))
    if not sequence_file.exists():
        print('NotFoundError: %s' % sequence_file)
        return []

    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    cur_save_dir = data_save_path / sequence_name
    cur_save_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = cur_save_dir / ('%s.pkl' % sequence_name)

    sequence_infos = []
    if pkl_file.exists():
        sequence_infos = pickle.load(open(pkl_file, 'rb'))
        sequence_infos_old = None
        if not update_info_only:
            print('Skip sequence since it has been processed before: %s' % pkl_file)
            return sequence_infos
        else:
            sequence_infos_old = sequence_infos
            sequence_infos = []

    for cnt, data in enumerate(dataset):
        if cnt % sampled_interval != 0:
            continue
        # print(sequence_name, cnt)
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        info = {}
        pc_info = {'num_features': 5, 'lidar_sequence': sequence_name, 'sample_idx': cnt}
        info['point_cloud'] = pc_info

        info['frame_id'] = sequence_name + ('_%03d' % cnt)
        info['metadata'] = {
            'context_name': frame.context.name,
            'timestamp_micros': frame.timestamp_micros
        }
        image_info = {}
        for j in range(5):
            width = frame.context.camera_calibrations[j].width
            height = frame.context.camera_calibrations[j].height
            image_info.update({'image_shape_%d' % j: (height, width)})
        info['image'] = image_info

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        info['pose'] = pose

        if has_label:
            annotations = generate_labels(frame, pose=pose)
            info['annos'] = annotations

        if update_info_only and sequence_infos_old is not None:
            assert info['frame_id'] == sequence_infos_old[cnt]['frame_id']
            num_points_of_each_lidar = sequence_infos_old[cnt]['num_points_of_each_lidar']
        else:
            num_points_of_each_lidar = save_lidar_points(
                frame, cur_save_dir / ('%04d.npy' % cnt), use_two_returns=use_two_returns
            )
        info['num_points_of_each_lidar'] = num_points_of_each_lidar

        sequence_infos.append(info)

    with open(pkl_file, 'wb') as f:
        pickle.dump(sequence_infos, f)

    print('Infos are saved to (sampled_interval=%d): %s' % (sampled_interval, pkl_file))
    return sequence_infos

def get_infos(sample_seq_list, data_root_path, data_save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1, update_info_only=False):
    from functools import partial

    process_single_sequence_func = partial(
        process_single_sequence,
        data_save_path=data_save_path, sampled_interval=sampled_interval, has_label=has_label, update_info_only=update_info_only
    )
    
    sample_sequence_file_list = [
        dataset.check_sequence_name_with_all_version(data_root_path / sequence_file)
        for sequence_file in sample_seq_list
    ]

    with multiprocessing.Pool(num_workers) as p:
        sequence_infos = list(tqdm.tqdm(p.imap(process_single_sequence_func, sample_sequence_file_list),
                                    total=len(sample_sequence_file_list)))

    all_sequences_infos = [item for infos in sequence_infos for item in infos]
    return all_sequences_infos


def create_groundtruth_database(data_root_path, save_path, info_path=None, used_classes=None, split='train', sampled_interval=10, processed_data_tag=None):
    import torch
    # TODO multiframe not supported
    use_sequence_data = SEQUENCE_CONFIG = None
    if use_sequence_data:
        st_frame, ed_frame = SEQUENCE_CONFIG.SAMPLE_OFFSET[0], SEQUENCE_CONFIG.SAMPLE_OFFSET[1]
        SEQUENCE_CONFIG.SAMPLE_OFFSET[0] = min(-4, st_frame)  # at least we use 5 frames for generating gt database to support various sequence configs (<= 5 frames)
        st_frame = SEQUENCE_CONFIG.SAMPLE_OFFSET[0]
        database_save_path = save_path / ('%s_gt_database_%s_sampled_%d_multiframe_%s_to_%s' % (processed_data_tag, split, sampled_interval, st_frame, ed_frame))
        db_info_save_path = save_path / ('%s_waymo_dbinfos_%s_sampled_%d_multiframe_%s_to_%s.pkl' % (processed_data_tag, split, sampled_interval, st_frame, ed_frame))
        db_data_save_path = save_path / ('%s_gt_database_%s_sampled_%d_multiframe_%s_to_%s_global.npy' % (processed_data_tag, split, sampled_interval, st_frame, ed_frame))
    else:
        database_save_path = save_path / ('%s_gt_database_%s_sampled_%d' % (processed_data_tag, split, sampled_interval))
        db_info_save_path = save_path / ('%s_waymo_dbinfos_%s_sampled_%d.pkl' % (processed_data_tag, split, sampled_interval))
        db_data_save_path = save_path / ('%s_gt_database_%s_sampled_%d_global.npy' % (processed_data_tag, split, sampled_interval))

    database_save_path.mkdir(parents=True, exist_ok=True)
    all_db_infos = {}

    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    point_offset_cnt = 0
    stacked_gt_points = []
    for k in tqdm.tqdm(range(0, len(infos), sampled_interval)):
        info = infos[k]
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        points = dataset.get_lidar(data_root_path, sequence_name, sample_idx)
        if use_sequence_data:
            points, num_points_all, sample_idx_pre_list, _, _, _, _ = dataset.get_sequence_data(
                info, points, sequence_name, sample_idx, SEQUENCE_CONFIG
            )
        annos = info['annos']
        names = annos['name']
        difficulty = annos['difficulty']
        gt_boxes = annos['gt_boxes_lidar']

        if k % 4 != 0 and len(names) > 0:
            mask = (names == 'Vehicle')
            names = names[~mask]
            difficulty = difficulty[~mask]
            gt_boxes = gt_boxes[~mask]

        if k % 2 != 0 and len(names) > 0:
            mask = (names == 'Pedestrian')
            names = names[~mask]
            difficulty = difficulty[~mask]
            gt_boxes = gt_boxes[~mask]

        num_obj = gt_boxes.shape[0]
        if num_obj == 0:
            continue
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
            torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
            torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
        ).long().squeeze(dim=0).cpu().numpy()

        for i in range(num_obj):
            filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
            filepath = database_save_path / filename
            gt_points = points[box_idxs_of_pts == i]
            gt_points[:, :3] -= gt_boxes[i, :3]

            if (used_classes is None) or names[i] in used_classes:
                gt_points = gt_points.astype(np.float32)
                assert gt_points.dtype == np.float32
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                db_path = str(filepath.relative_to(save_path))  # gt_database/xxxxx.bin
                db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                            'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                            'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}

                # it will be used if you choose to use shared memory for gt sampling
                stacked_gt_points.append(gt_points)
                db_info['global_data_offset'] = [point_offset_cnt, point_offset_cnt + gt_points.shape[0]]
                point_offset_cnt += gt_points.shape[0]

                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
    for k, v in all_db_infos.items():
        print('Database %s: %d' % (k, len(v)))

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)

    # it will be used if you choose to use shared memory for gt sampling
    stacked_gt_points = np.concatenate(stacked_gt_points, axis=0)
    np.save(db_data_save_path, stacked_gt_points)

def create_waymo_infos(data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='waymo_processed_data', update_info_only=False,
                       ):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    split_dict = {
        'train': 'ImageSets/train.txt', 
        'val': 'ImageSets/val.txt', 
        }
    for split, split_file in split_dict.items():
        save_filename = save_path / f"{processed_data_tag}_infos_{split}.pkl"
        
        sample_seq_list = [x.strip() for x in open(save_path / split_file).readlines()]

        infos = get_infos(sample_seq_list, data_root_path=data_path / raw_data_tag,
            data_save_path=data_path / processed_data_tag, has_label=True, 
            sampled_interval=1, update_info_only=update_info_only)
        with open(save_filename, 'wb') as f:
            pickle.dump(infos, f)
        print(f'waymo info {split} file is saved to {save_filename}')

        if split == 'train' and not update_info_only:
            print('Start create groundtruth database for data augmentation')
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            create_groundtruth_database(data_path / processed_data_tag, save_path, info_path=save_filename, split=split,
                sampled_interval=1, used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=processed_data_tag)
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    import argparse
    from pcdet.config import cfg

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    parser.add_argument('--processed_data_tag', type=str, default='waymo_processed_data_v1_4_0', help='dataset version v1_4_0 updated in 2022')
    parser.add_argument('--update_info_only', action='store_true', default=False, help='')

    args = parser.parse_args()

    if args.func == 'create_waymo_infos':
        # TODO remove create_groundtruth_database_parallel
        create_waymo_infos(
            data_path=cfg.ROOT_DIR / 'data' / 'waymo',
            save_path=cfg.ROOT_DIR / 'data' / 'waymo',
            raw_data_tag='archived_files',
            processed_data_tag=args.processed_data_tag,
            update_info_only=args.update_info_only
        )
    else:
        raise NotImplementedError
