import numpy as np
from kmn.util import trans_to_scaling, trans_scaling, trans_xya, trans_to_xya


def trans_to_param(T):
    """
    :param T: transformation matrix

    :return param: parameter of the scene
    """
    sx, sy, sz = trans_to_scaling(T)
    x, y, theta = trans_to_xya(T)
    param = np.array([x, y, theta, sy-1.0, sz-1.0])
    return param


def param_to_trans(param):
    """
    :param param: parameter of the scene

    :return T: transformation matrix
    """
    if param.ndim > 1:
        param = np.squeeze(param)
    x = param[0]
    y = param[1]
    theta = param[2]
    sy = param[3]
    sz = param[4]

    T_1 = trans_xya(x=x, y=y, angle=theta)
    T_2 = trans_scaling(sy=1.0 + sy, sz=1.0 + sz)
    T = np.matmul(T_1, T_2)

    return T
