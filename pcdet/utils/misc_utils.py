import numpy as np

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
