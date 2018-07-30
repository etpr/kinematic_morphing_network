import numpy as np
from kmn.util import trans_to_xya, trans_xya


def trans_to_param(T):
    """
    :param T: transformation matrix

    :return param: parameter of the scene
    """
    x, y, theta = trans_to_xya(T)
    param = np.array([x, y])
    return param


def param_to_trans(param):
    """
    :param param: parameter of the scene

    :return T: transformation matrix
    """
    if param.ndim > 1:
        param = np.squeeze(param)
    T = trans_xya(param[0], param[1], 0)
    return T
