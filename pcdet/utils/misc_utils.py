import numpy as np
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import torch

def transform_matrix(rotation_mat, translation, inverse: bool = False) -> np.ndarray:
    """
    返回变换矩阵或变换矩阵的逆，直接对变换矩阵求逆可能无解报错
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation_mat.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation_mat
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm

def replace_gt(raw_points, raw_boxes3d, mix_points, mix_boxes3d):
    raw_point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(raw_points[:, 0:3]), torch.from_numpy(raw_boxes3d[:, :7])
    ).numpy()
    mix_point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(mix_points[:, 0:3]), torch.from_numpy(mix_boxes3d[:, :7])
    ).numpy()
    points = raw_points[np.sum(raw_point_indices, axis=0) == 0]
    points = np.vstack((mix_points[np.sum(mix_point_indices, axis=0) > 0], points))
    return points
