import numpy as np
from kmn.util import transform_pointcloud
from PIL import Image, ImageFilter


def f(P, T_c_w, cam_param):
    """
    Renders a depth image from a point cloud and camera properties
    :param P: point cloud
    :param T_c_w: transformation matrix from camera to world frame
    :param cam_param: intrinsic camera parameters

    :return D : rendered depth image
    """
    fovy = cam_param[0]
    aspect = cam_param[1]
    zFar = cam_param[2]
    zNear = cam_param[3]
    focalLength = cam_param[4]
    d_min = cam_param[5]

    # transform pointcloud to camera frame
    pcT = transform_pointcloud(T_c_w, P)

    # perspective projection
    pcT[:, 0] = (2.0 * focalLength / aspect * pcT[:, 0] / pcT[:, 2] + 1.0) * 320.0
    pcT[:, 1] = (2.0 * focalLength * pcT[:, 1] / pcT[:, 2] + 1.0) * 240.0
    pcT[:, 2] = -pcT[:, 2]

    # compute depth value
    c1 = (pcT[:, 2] - zNear) / (zFar - zNear)
    d = (c1 * zFar / zNear + c1) / (1.0 + c1 * zFar / zNear)
    d = (d-d_min)/(1.0-d_min)
    d *= 255.0
    d = np.uint8(np.floor(d))
    d[d == 254] = 255

    # reformat as Depth image
    D = np.ones((640, 480), dtype=np.uint8) * 255

    C_idx = np.empty(shape=(pcT.shape[0], 2), dtype=int)
    C_idx[:, 0] = np.round(pcT[:, 0])
    C_idx[:, 1] = np.round(pcT[:, 1])

    c1 = np.greater_equal(C_idx[:, 0], 0)
    c2 = np.greater_equal(C_idx[:, 1], 0)
    c3 = np.less(C_idx[:, 0], 640)
    c4 = np.less(C_idx[:, 1], 480)
    c = np.logical_and(np.logical_and(np.logical_and(c1, c2), c3), c4)

    idx_sort = np.flip(np.argsort(d), axis=0)

    idx_sort = idx_sort[c[idx_sort]]

    for i in range(idx_sort.size):
        j = idx_sort[i]
        D[C_idx[j, 0], C_idx[j, 1]] = d[j]

    # rotate image -90 degree
    D = D[::-1, :]
    D = np.transpose(D)

    # remove small holes that occur through translation
    D = Image.fromarray(D)
    im_filt = ImageFilter.MinFilter(size=5)
    D = D.filter(im_filt)
    im_filt = ImageFilter.MaxFilter(size=5)
    D = D.filter(im_filt)
    D = np.array(D)

    return D
