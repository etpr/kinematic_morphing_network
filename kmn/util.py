from __future__ import print_function
import numpy as np
import os
import sys
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def axis_equal(ax):
    """
    Equally align axes of a 3D plot
    :param ax: matplotlib axis
    """
    x_lim = ax.get_xlim3d()
    y_lim = ax.get_ylim3d()
    z_lim = ax.get_zlim3d()

    r = 0.5*max([abs(x_lim[1] - x_lim[0]), abs(y_lim[1] - y_lim[0]), abs(z_lim[1] - z_lim[0])])
    x_c = np.mean(x_lim)
    y_c = np.mean(y_lim)
    z_c = np.mean(z_lim)

    ax.set_xlim3d([x_c - r, x_c + r])
    ax.set_ylim3d([y_c - r, y_c + r])
    ax.set_zlim3d([z_c - r, z_c + r])


def conv_quat_to_axis_angle(q):
    """
    Converts a quaternion to axis-angle representation
    https://en.wikipedia.org/wiki/Axis-angle_representation#Unit_quaternions
    :param q: quaternion

    :return axis: rotation axis
    :return alpha: rotation angle in radian
    """
    if not np.isclose(np.linalg.norm(q), 1.0):
        q /= np.linalg.norm(q)

    alpha = 2.0 * np.arctan2(np.linalg.norm(q[0:3], ord=2), q[3])
    if alpha == 0.0:
        axis = np.zeros(3)
    else:
        axis = q[0:3] / (np.sin(alpha / 2.0))

    return axis, alpha


def conv_axis_angle_to_quat(axis, alpha):
    """
    Convert rotation axis and angle to quaternion
    :param axis: rotation axis
    :param alpha: rotation angle in radian

    :return q: quaternion
    """
    if not (np.isclose(axis, 0)).all():
        axis = axis / np.linalg.norm(axis)
    tmp = np.sin(alpha / 2.0)
    q = [tmp * axis[0], tmp * axis[1], tmp * axis[2], np.cos(alpha / 2.0)]
    return q


def conv_quat_to_rot_vector(q):
    """
    Covert quaternion to rotation vector
    :param q: quaternion

    :return axis: rotation vector
    """
    axis, alpha = conv_quat_to_axis_angle(q)
    axis = axis * alpha
    return axis


def conv_quat_to_matrix(q):
    """
    Convert quaternion to rotation matrix
    :param q: quaternion

    :return R: rotation matrix R
    """
    q = q/np.linalg.norm(q)
    R = np.zeros(shape=(3, 3))
    R[0, 0] = 1 - 2 * q[1] * q[1] - 2 * q[2] * q[2]
    R[1, 1] = 1 - 2 * q[0] * q[0] - 2 * q[2] * q[2]
    R[2, 2] = 1 - 2 * q[0] * q[0] - 2 * q[1] * q[1]
    R[0, 1] = 2 * q[0] * q[1] - 2 * q[3] * q[2]
    R[1, 0] = 2 * q[0] * q[1] + 2 * q[3] * q[2]
    R[0, 2] = 2 * q[0] * q[2] + 2 * q[3] * q[1]
    R[2, 0] = 2 * q[0] * q[2] - 2 * q[3] * q[1]
    R[1, 2] = 2 * q[1] * q[2] - 2 * q[3] * q[0]
    R[2, 1] = 2 * q[1] * q[2] + 2 * q[3] * q[0]

    return R


def conv_matrix_to_quat(R):
    """
    Convert rotation matrix to quaternion
    :param R: rotation matrix

    :return q: quaternion
    """
    q = np.zeros(shape=4)
    print(np.trace(R))
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.) * 2
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
        q[3] = 0.25 * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q[0] = 0.25 * s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = (R[0, 2] + R[2, 0]) / s
        q[3] = (R[2, 1] - R[1, 2]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q[0] = (R[0, 1] + R[1, 0]) / s
        q[1] = 0.25 * s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = (R[0, 2] - R[2, 0]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q[0] = (R[0,2] + R[2, 0]) / s
        q[1] = (R[1, 2] + R[2, 1]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 0] - R[0, 1]) / s
    return q


def progress_bar(i, N):
    """
    Prints a prograss bar on the terminal
    :param i: current iteration between 0 and [N-1]
    :param N: total number of iterations N
    """
    if N == 1:
        return
    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %d%%" % ('.' * np.int32(i / (N-1) * 100), np.int32(i / (N-1) * 100)))
    if i == (N-1):
        sys.stdout.write('\n')
    sys.stdout.flush()


def create_folder(name):
    """
    Creates a directory if it does not exist yet
    :param name: directory path
    """
    if not os.path.exists(name):
        os.makedirs(name)


def classification_stats(y, y_pred, print_stats=True, file=""):
    """
    Calculates statistics of a classification problem
    :param y: ground truth class
    :param y_pred: predicted class
    :param print_stats: option to print stats in terminal
    :param file: prints statistics to text file

    :returns: accuracy, precision, recall, f1
    """
    true_pos = np.float(np.count_nonzero(y[y == 1] == y_pred[y == 1]))
    true_neg = np.float(np.count_nonzero(y[y == 0] == y_pred[y == 0]))
    false_pos = np.float(np.count_nonzero(y_pred[y_pred == 1] != y[y_pred == 1]))
    false_neg = np.float(np.count_nonzero(y_pred[y_pred == 0] != y[y_pred == 0]))

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2.0 * precision * recall / (precision + recall)
    accuracy = (true_pos+true_neg) / np.float(y.shape[0])

    if print_stats:
        print("true_positives: ", true_pos)
        print("true_negatives: ", true_neg)
        print("false_positives: ", false_pos)
        print("false_negatives: ", false_neg)
        print("precision: ", precision)
        print("accuracy: ", accuracy)
        print("recall: ", recall)
        print("f1: ", f1)

    if file is not "":
        f = open(file, 'w')
        print("true_positives: ", true_pos, file=f)
        print("true_negatives: ", true_neg, file=f)
        print("false_positives: ", false_pos, file=f)
        print("false_negatives: ", false_neg, file=f)
        print("precision: ", precision, file=f)
        print("accuracy: ", accuracy, file=f)
        print("recall: ", recall, file=f)
        print("f1: ", f1, file=f)
        f.close()

    return accuracy, precision, recall, f1


def regression_stats(y, y_pred, print_stats=True, file=""):
    """
    Calculates statistics of a regression problem
    :param y: ground truth value
    :param y_pred: predicted value
    :param print_stats: print metrics on terminal
    :param file: prints metrics to text file

    :returns: return mae, mae_std, sign_rate
    """
    res = y - y_pred
    mae = np.mean(np.abs(res))
    mae_std = np.std(np.abs(res))
    sign_rate = np.sum(np.equal(np.sign(y_pred), np.sign(y))) / np.double(y.shape[0])

    if print_stats:
        print("mae: ", mae)
        print("mae_std: ", mae_std)
        print("sign_rate: ", sign_rate)

    if file is not "":
        f = open(file, 'w')
        print("mae: ", mae, file=f)
        print("mae_std: ", mae_std, file=f)
        print("sign_rate: ", sign_rate, file=f)
        f.close()

    return mae, mae_std, sign_rate


def comp_angle_between_vectors(vec1, vec2):
    """
    Computes the angle between two vectors in radian
    :param vec1: first vector
    :param vec2: second vector

    :return alpha: angle in radian
    """
    tmp = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if np.isclose(tmp, 0.0):
        alpha = 0.0
    else:
        tmp = np.dot(vec1, vec2) / tmp
        tmp = np.min([tmp, 1.0])
        tmp = np.max([tmp, -1.0])
        alpha = np.arccos(tmp)
    return alpha


def transform_pointcloud(trans, P):
    """
    :param trans: (4,4) transformation matrix
    :param P: (:, 3) point cloud

    :return P_trans: (:,3) transformed point cloud
    """

    if trans.shape != (4, 4):
        raise TypeError("Transformaiton matrix must be of size (4,4)")
    if P.shape[1] != 3:
        raise TypeError("pointcloud must have 3 columns")

    P = np.matmul(np.concatenate((P, np.ones((P.shape[0], 1))), axis=1), np.transpose(trans))
    P = P[:, 0:3]
    return P


def invert_trans(T):
    """
    Inverts an affine transformation matrix
    :param T: (4,4) transformation matrix

    :return T_inv: (4,4) inverted transformation matrix
    """
    if T.shape != (4, 4):
        return TypeError("Transformation matrix must be 4 x 4")

    R = T[0:3, 0:3]
    t = T[0:3, 3]
    T_inv = np.eye(4)
    T_inv[0:3, 0:3] = np.linalg.inv(R)
    T_inv[0:3, 3] = -np.matmul(np.linalg.inv(R), t)
    return T_inv


def trans_xya(x, y, angle):
    """
    Create a transformation matrix for a x and y translation as well as a rotation around the z
    """
    c, s = np.cos(angle), np.sin(angle)
    R = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
    t = np.array([x, y, 0.])
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


def trans_to_xya(T):
    """
    Extract translation along x and y as well as a rotation around z of a transformation matrix
    """
    x = T[0, 3]
    y = T[1, 3]
    angle = np.arctan2(T[1, 0], T[0, 0])
    return x, y, angle


def trans_to_scaling(T):
    """
    Extract xyz scales of a transformation matrix
    """
    sx = np.linalg.norm(T[0:3, 0])
    sy = np.linalg.norm(T[0:3, 1])
    sz = np.linalg.norm(T[0:3, 2])
    return sx, sy, sz


def trans_rot(x_angle=0, y_angle=0, z_angle=0):
    """
    Create a transformation matrix for a rotation around x, y, and z
    """
    Tx = trans_rot_x(x_angle)
    Ty = trans_rot_y(y_angle)
    Tz = trans_rot_z(z_angle)
    T = np.matmul(Tx, np.matmul(Ty, Tz))
    return T


def trans_rot_x(angle):
    """
    Create a transformation matrix for a rotation around the x axis
    """
    T = np.eye(4)
    T[1, 1] = T[2, 2] = np.cos(angle)
    T[2, 1] = np.sin(angle)
    T[1, 2] = -np.sin(angle)
    return T


def trans_rot_y(angle):
    """
    Create a transformation matrix for a rotation around the y axis
    """
    T = np.eye(4)
    T[0, 0] = T[2, 2] = np.cos(angle)
    T[2, 0] = -np.sin(angle)
    T[0, 2] = np.sin(angle)
    return T


def trans_rot_z(angle):
    """
    Create a transformation matrix for a rotation around the z axis
    """
    T = np.eye(4)
    T[0, 0] = T[1, 1] = np.cos(angle)
    T[1, 0] = np.sin(angle)
    T[0, 1] = -np.sin(angle)
    return T


def trans_scaling(sx=1.0, sy=1.0, sz=1.0):
    """
    Create a transformation matrix from xyz scales
    """
    T = np.eye(4)
    T[0, 0] = sx
    T[1, 1] = sy
    T[2, 2] = sz
    return T


def trans_R_t(R=np.eye(3), t=np.zeros(3)):
    """
    Create a transformation matrix from rotation matrix and translation vector
    """
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


def trans_translation(tx=0.0, ty=0.0, tz=0.0):
    """
    Create a translation vector
    """
    T = np.eye(4)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def resize_image(D, w, h):
    """
    Resize an image D to width w and height h
    """
    if type(D) is np.ndarray:
        D = Image.fromarray(D)
    D = D.resize((w, h), Image.BICUBIC)
    D = np.array(D)
    return D


def limit_value(value, max_value, min_value):
    """
    limits value to [min_value, max_value]
    """
    # value max and min are arrays
    if (type(value) is np.ndarray) and type(max_value) is np.ndarray and type(min_value) is np.ndarray:
        if (max_value < min_value).any():
            raise ValueError("max value is greater than min value")
        res = np.empty_like(value)
        for i in range(value.size):
            res.itemset(i, np.max([np.min([value.item(i), max_value.item(i)]), min_value.item(i)]))
        return res

    if max_value <= min_value:
        raise ValueError("max value is greater than min value")

    # value is array, max and min are floats
    if (type(value) is np.ndarray) and type(max_value) is float and type(min_value) is float:
        res = np.empty_like(value)
        for i in range(value.size):
            res.itemset(i, np.max([np.min([value.item(i), max_value]), min_value]))
        return res

    # value, max, min are floats
    if type(value) is float:
        return np.max([np.min([value, max_value]), min_value])

    raise TypeError("inputs to limit_value have to be arrays or floats")


def check_limits(value, max_value, min_value):
    """
    check if a value is between [min_value, max_value]
    """
    # value, max and min are arrays
    if (type(value) is np.ndarray) and type(max_value) is np.ndarray and type(min_value) is np.ndarray:
        for i in range(value.size):
            if value.item(i) > max_value.item(i) or value.item(i) < min_value.item(i):
                return False
        return True

    # value is array, max and min are floats
    if (type(value) is np.ndarray) and type(max_value) is float and type(min_value) is float:
        for i in range(value.size):
            if value.item(i) > max_value or value.item(i) < min_value:
                return False
        return True

    # value, max, min are floats
    if type(value) is float:
        return value <= max_value and value >= min_value

    raise TypeError("inputs to limit_value have to be arrays or floats")


def error_mae(x1, x2):
    """
    computes the mean absolute error between two vectors
    """
    return np.sum(np.abs(x1-x2))/np.double(x1.size)


def error_mse(x1, x2):
    """
    computes the mean squared error between two vectors
    """
    return np.mean(np.square(x1 - x2), axis=1)


def get_col_idx(i):
    """
    define some colors for plotting
    """
    cols = 'rkgbmckrgbmckrkgbmckrgbmckrkgbmckrgbmck'
    return cols[i]


def split_test_train_data(x, y, ratio=0.8):
    """
    split data (x, y) into a train and test set for a given ratio
    """
    n = x.shape[0]
    x_test = x[int(ratio * n):, :]
    y_test = y[int(ratio * n):, :]
    x_train = x[:int(ratio * n), :]
    y_train = y[:int(ratio * n), :]
    return x_train, y_train, x_test, y_test


def plot_pointcloud(P, fig=None, c='b', sub_sample=1):
    """
    creates a 3D point cloud plot (with optional sub_sample)
    """
    if fig is None:
        fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(P[::sub_sample, 0], P[::sub_sample, 1], P[::sub_sample, 2], c=c)
    return fig


def subsample_pointcloud(P, sub_sample):
    """
    reduce size of point cloud by subsampling
    """
    return P[0::sub_sample, :]