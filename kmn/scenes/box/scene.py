import numpy as np
from kmn.util import trans_to_scaling, trans_scaling


def trans_to_param(T):
    """
    :param T: transformation matrix

    :return param: parameter of the scene
    """
    sx, sy, sz = trans_to_scaling(T)
    param = np.array([sx-1.0])
    return param


def param_to_trans(param):
    """
    :param param: parameter of the scene

    :return T: transformation matrix
    """
    if param.ndim > 1:
        param = np.squeeze(param)
    T = trans_scaling(sx=1.0 + param, sy=1.0 + param)
    return T
