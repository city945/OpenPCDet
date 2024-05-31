"""
modified from ONCE eval
"""
import numpy as np
from pcdet.datasets.once.once_eval.eval_utils import compute_split_parts, overall_filter, distance_filter
from pcdet.datasets.once.once_eval.evaluation import compute_iou3d, accumulate_scores, get_thresholds, compute_statistics

iou_threshold_dict = {
    'Car': 0.7,
    'Truck': 0.7,
    'Pedestrian': 0.3,
    'Bicyclist': 0.5,
    'Motorcyclist': 0.5
}

superclass_iou_threshold_dict = {
    'Vehicle': 0.7,
    'Pedestrian': 0.3,
    'Cyclist': 0.5
}

def get_evaluation_results(gt_annos, pred_annos, classes,
                           use_superclass=False,
                           iou_thresholds=None,
                           num_pr_points=50,
                           difficulty_mode='Distance70',
                           ap_with_heading=True,
                           num_parts=100,
                           print_ok=False
                           ):

    if iou_thresholds is None:
        if use_superclass:
            iou_thresholds = superclass_iou_threshold_dict
        else:
            iou_thresholds = iou_threshold_dict

    assert len(gt_annos) == len(pred_annos), "the number of GT must match predictions"
    assert difficulty_mode in ['Overall&Distance', 'Overall', 'Distance', 'Distance70'], "difficulty mode is not supported"
    if use_superclass:
        mapped_classes = False
        for cls_name in classes:
            if cls_name == 'Pedestrian':
                mapped_classes.append('Pedestrian')
            if cls_name in ['Car', 'Truck']:
                if 'Vehicle' not in mapped_classes:
                    mapped_classes.append('Vehicle')
            elif cls_name in ['Bicyclist', 'Motorcyclist']:
                if 'Cyclist' not in mapped_classes:
                    mapped_classes.append('Cyclist')
            else:
                raise ValueError(f"find undefined class name {cls_name}")
        classes = mapped_classes
 
    num_samples = len(gt_annos)
    split_parts = compute_split_parts(num_samples, num_parts)
    ious = compute_iou3d(gt_annos, pred_annos, split_parts, with_heading=ap_with_heading)

    num_classes = len(classes)
    if difficulty_mode == 'Distance':
        num_difficulties = 3
        difficulty_types = ['0-30m', '30-50m', '50m-inf']
    elif difficulty_mode == 'Overall':
        num_difficulties = 1
        difficulty_types = ['overall']
    elif difficulty_mode == 'Distance70':
        num_difficulties = 1
        difficulty_types = ['0-70m']
    elif difficulty_mode == 'Overall&Distance':
        num_difficulties = 4
        difficulty_types = ['overall', '0-30m', '30-50m', '50m-inf']
    else:
        raise NotImplementedError

    precision = np.zeros([num_classes, num_difficulties, num_pr_points+1])
    recall = np.zeros([num_classes, num_difficulties, num_pr_points+1])

    for cls_idx, cur_class in enumerate(classes):
        iou_threshold = iou_thresholds[cur_class]
        for diff_idx in range(num_difficulties):
            ### filter data & determine score thresholds on p-r curve ###
            accum_all_scores, gt_flags, pred_flags = [], [], []
            num_valid_gt = 0
            for sample_idx in range(num_samples):
                gt_anno = gt_annos[sample_idx]
                pred_anno = pred_annos[sample_idx]
                pred_score = pred_anno['score']
                iou = ious[sample_idx]
                gt_flag, pred_flag = filter_data(gt_anno, pred_anno, difficulty_mode,
                                                    difficulty_level=diff_idx, class_name=cur_class, use_superclass=use_superclass)
                gt_flags.append(gt_flag)
                pred_flags.append(pred_flag)
                num_valid_gt += sum(gt_flag == 0)
                accum_scores = accumulate_scores(iou, pred_score, gt_flag, pred_flag,
                                                 iou_threshold=iou_threshold)
                accum_all_scores.append(accum_scores)
            all_scores = np.concatenate(accum_all_scores, axis=0)
            thresholds = get_thresholds(all_scores, num_valid_gt, num_pr_points=num_pr_points)

            ### compute tp/fp/fn ###
            confusion_matrix = np.zeros([len(thresholds), 3]) # only record tp/fp/fn
            for sample_idx in range(num_samples):
                pred_score = pred_annos[sample_idx]['score']
                iou = ious[sample_idx]
                gt_flag, pred_flag = gt_flags[sample_idx], pred_flags[sample_idx]
                for th_idx, score_th in enumerate(thresholds):
                    tp, fp, fn = compute_statistics(iou, pred_score, gt_flag, pred_flag,
                                                    score_threshold=score_th, iou_threshold=iou_threshold)
                    confusion_matrix[th_idx, 0] += tp
                    confusion_matrix[th_idx, 1] += fp
                    confusion_matrix[th_idx, 2] += fn

            ### draw p-r curve ###
            for th_idx in range(len(thresholds)):
                recall[cls_idx, diff_idx, th_idx] = confusion_matrix[th_idx, 0] / \
                                                    (confusion_matrix[th_idx, 0] + confusion_matrix[th_idx, 2])
                precision[cls_idx, diff_idx, th_idx] = confusion_matrix[th_idx, 0] / \
                                                       (confusion_matrix[th_idx, 0] + confusion_matrix[th_idx, 1])

            for th_idx in range(len(thresholds)):
                precision[cls_idx, diff_idx, th_idx] = np.max(
                    precision[cls_idx, diff_idx, th_idx:], axis=-1)
                recall[cls_idx, diff_idx, th_idx] = np.max(
                    recall[cls_idx, diff_idx, th_idx:], axis=-1)

    AP = 0
    for i in range(1, precision.shape[-1]):
        AP += precision[..., i]
    AP = AP / num_pr_points * 100

    ret_dict = {}

    ret_str = "\n|AP@%-9s|" % (str(num_pr_points))
    for diff_type in difficulty_types:
        ret_str += '%-12s|' % diff_type
    ret_str += '\n'
    for cls_idx, cur_class in enumerate(classes):
        ret_str += "|%-12s|" % cur_class
        for diff_idx in range(num_difficulties):
            diff_type = difficulty_types[diff_idx]
            key = 'AP_' + cur_class + '/' + diff_type
            ap_score = AP[cls_idx,diff_idx]
            ret_dict[key] = ap_score
            ret_str += "%-12.2f|" % ap_score
        ret_str += "\n"
    mAP = np.mean(AP, axis=0)
    ret_str += "|%-12s|" % 'mAP'
    for diff_idx in range(num_difficulties):
        diff_type = difficulty_types[diff_idx]
        key = 'AP_mean' + '/' + diff_type
        ap_score = mAP[diff_idx]
        ret_dict[key] = ap_score
        ret_str += "%-12.2f|" % ap_score
    ret_str += "\n"

    if print_ok:
        print(ret_str)
    return ret_str, ret_dict

def filter_data(gt_anno, pred_anno, difficulty_mode, difficulty_level, class_name, use_superclass):
    """
    Filter data by class name and difficulty

    Args:
        gt_anno:
        pred_anno:
        difficulty_mode:
        difficulty_level:
        class_name:

    Returns:
        gt_flags/pred_flags:
            1 : same class but ignored with different difficulty levels
            0 : accepted
           -1 : rejected with different classes
    """
    num_gt = len(gt_anno['name'])
    gt_flag = np.zeros(num_gt, dtype=np.int64)
    if use_superclass:
        if class_name == 'Vehicle':
            reject = np.logical_not(np.logical_or(gt_anno['name']=='Car', gt_anno['name']=='Truck'))
        elif class_name == 'Cyclist':
            reject = np.logical_not(np.logical_or(gt_anno['name']=='Bicyclist', gt_anno['name']=='Motorcyclist'))
        else:
            reject = gt_anno['name'] != class_name
    else:
        reject = gt_anno['name'] != class_name
    gt_flag[reject] = -1
    num_pred = len(pred_anno['name'])
    pred_flag = np.zeros(num_pred, dtype=np.int64)
    if use_superclass:
        if class_name == 'Vehicle':
            reject = np.logical_not(np.logical_or(gt_anno['name']=='Car', gt_anno['name']=='Truck'))
        elif class_name == 'Cyclist':
            reject = np.logical_not(np.logical_or(gt_anno['name']=='Bicyclist', gt_anno['name']=='Motorcyclist'))
        else:
            reject = pred_anno['name'] != class_name
    else:
        reject = pred_anno['name'] != class_name
    pred_flag[reject] = -1

    if difficulty_mode == 'Overall':
        ignore = overall_filter(gt_anno['gt_boxes_lidar'])
        gt_flag[ignore] = 1
        ignore = overall_filter(pred_anno['boxes_lidar'])
        pred_flag[ignore] = 1
    elif difficulty_mode == 'Distance':
        ignore = distance_filter(gt_anno['gt_boxes_lidar'], difficulty_level)
        gt_flag[ignore] = 1
        ignore = distance_filter(pred_anno['boxes_lidar'], difficulty_level)
        pred_flag[ignore] = 1
    elif difficulty_mode == 'Overall&Distance':
        ignore = overall_distance_filter(gt_anno['gt_boxes_lidar'], difficulty_level)
        gt_flag[ignore] = 1
        ignore = overall_distance_filter(pred_anno['boxes_lidar'], difficulty_level)
        pred_flag[ignore] = 1
    elif difficulty_mode == 'Distance70':
        ignore = overall_distance_filter(gt_anno['gt_boxes_lidar'], difficulty_level, difficulty_mode)
        gt_flag[ignore] = 1
        ignore = overall_distance_filter(pred_anno['boxes_lidar'], difficulty_level, difficulty_mode)
        pred_flag[ignore] = 1
    else:
        raise NotImplementedError

    return gt_flag, pred_flag

def overall_distance_filter(boxes, level, mode=None):
    ignore = np.ones(boxes.shape[0], dtype=bool)  # all true
    dist = np.sqrt(np.sum(boxes[:, 0:3] * boxes[:, 0:3], axis=1))

    if mode == 'Distance70' and level == 0:
        flag = dist < 70
    elif level == 0:
        flag = np.ones(boxes.shape[0], dtype=bool)
    elif level == 1:  # 0-30m
        flag = dist < 30
    elif level == 2:  # 30-50m
        flag = (dist >= 30) & (dist < 50)
    elif level == 3:  # 50m-inf
        flag = dist >= 50
    else:
        assert False, 'level < 4 for overall & distance metric, found level %s' % (str(level))

    ignore[flag] = False
    return ignore
