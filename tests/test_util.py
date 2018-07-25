from kmn.util import *
import matplotlib.pyplot as plt
import pytest
from definitions import ROOT_DIR

VIS_TEST = False

d = 1./np.sqrt(2.)
q_axis_rot = [
    (np.array([0.0, 0.0, 0.0, 1.0]), np.array([0., 0., 0.]), 0),
    (np.array([1.0, 0.0, 0.0, 0.0]), np.array([1., 0., 0.]), np.pi),
    (np.array([0.0, 1.0, 0.0, 0.0]), np.array([0., 1., 0.]), np.pi),
    (np.array([0.0, 0.0, 1.0, 0.0]), np.array([0., 0., 1.]), np.pi),
    (np.array([d, 0.0, 0.0, d]), np.array([1., 0., 0.]), np.pi/2.),
    (np.array([0.0, d, 0.0, d]), np.array([0., 1., 0.]), np.pi/2.),
    (np.array([0.0, 0.0, d, d]), np.array([0., 0., 1.]), np.pi/2.),
    (np.array([.5, .5, .5, .5]), np.array([1., 1., 1.]/np.sqrt(3)), np.pi*2./3.),
]


@pytest.mark.parametrize("q, axis, rot", q_axis_rot)
def test_q_2_rot_axis_conv(q, axis, rot):
    axis2, rot2 = conv_quat_to_axis_angle(q)
    q2 = conv_axis_angle_to_quat(axis2, rot2)
    assert (np.isclose(axis, axis2)).all()
    assert np.isclose(rot, rot2)
    assert (np.isclose(q, q2)).all()


@pytest.mark.parametrize("q, axis, rot", q_axis_rot)
def test_rot_axis_2_q_conv(q, axis, rot):
    q2 = conv_axis_angle_to_quat(axis, rot)
    axis2, rot2 = conv_quat_to_axis_angle(q2)
    assert (np.isclose(axis, axis2)).all()
    assert np.isclose(rot, rot2)
    assert (np.isclose(q, q2)).all()


@pytest.mark.parametrize("q, axis, rot", q_axis_rot)
def test_rot_matrices_quat_conv(q, axis, rot):
    R = conv_quat_to_matrix(q)
    print(R)
    q2 = conv_matrix_to_quat(R)
    print(q2)
    assert np.isclose(q2, q).all()


def test_rot_matrices():
    axis = np.array([1.0, 0.0, 0.0])
    angle = 0.0
    q = conv_axis_angle_to_quat(axis, angle)
    R = conv_quat_to_matrix(q)
    assert (np.isclose(R, np.eye(3))).all()


def test_comp_angle_between_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    angle = comp_angle_between_vectors(v1, v2)
    assert angle == np.pi/2.0
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    angle = comp_angle_between_vectors(v1, v2)
    assert angle == 0.0
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    angle = comp_angle_between_vectors(v1, v2)
    assert angle == np.pi


def test_create_2d_transformation_matrix():
    x = np.random.rand()
    y = np.random.rand()
    theta = 2.0*(np.random.rand()-0.5)*np.pi
    T = trans_xya(x, y, theta)
    x2, y2, theta2 = trans_to_xya(T)
    assert x == x2
    assert y == y2
    assert np.isclose(theta, theta2)


def test_create_scaling_matrix():
    sx = np.random.rand()*2.0
    sy = np.random.rand()*2.0
    sz = np.random.rand()*2.0
    T = trans_scaling(sx, sy, sz)
    sx2, sy2, sz2 = trans_to_scaling(T)
    assert sx == sx2
    assert sy == sy2
    assert sz == sz2


def test_invert_trans():
    x = np.random.rand()
    y = np.random.rand()
    theta = 2.0*(np.random.rand()-0.5)*np.pi
    T = trans_xya(x, y, theta)
    T2 = trans_scaling(sx=1.2, sy=2.0, sz=0.5)
    T = np.matmul(T, T2)
    T_inv = invert_trans(T)
    assert (np.isclose(T, np.linalg.inv(T_inv))).all()
    T = trans_xya(x, y, 0.0)
    T_inv = invert_trans(T)
    x2, y2, theta2 = trans_to_xya(T_inv)
    assert x2 == -x
    assert y2 == -y
    T = trans_xya(0.0, 0.0, theta)
    T_inv = invert_trans(T)
    x2, y2, theta2 = trans_to_xya(T_inv)
    assert np.isclose(np.array(theta), np.array(-theta2))


def test_transform_pointcloud():
    x = np.random.rand()
    y = np.random.rand()
    theta = 2.0*(np.random.rand()-0.5)*np.pi
    T = trans_xya(x, y, 0.0)
    pc = np.random.randn(10, 3)
    pc_trans = transform_pointcloud(T, pc)
    pc_trans2 = pc + np.array([x, y, 0.0])
    assert (pc_trans == pc_trans2).all()

    T = trans_xya(x, y, theta)
    pc_trans = transform_pointcloud(T, pc)
    T_inv = np.linalg.inv(T)
    pc2 = transform_pointcloud(T_inv, pc_trans)
    assert np.isclose(pc2, pc).all()

    T = trans_scaling(sy=2.0)
    pc_trans = transform_pointcloud(T, pc)
    pc_trans2 = pc
    pc_trans2[:, 1] *= 2.0
    assert (pc_trans == pc_trans2).all()


def test_resize_image():
    D = Image.open(ROOT_DIR + '/kmn/scenes/box/data/raw/0_d.ppm')

    w = 128
    h = 96
    D_small = resize_image(D, w, h)

    assert D_small.shape[0] == h
    assert D_small.shape[1] == w

    if VIS_TEST:
        plt.figure(1)
        plt.imshow(D)
        plt.figure(2)
        plt.imshow(D_small)
        plt.show()


def test_limit_value():
    v = -2.5
    max_value = 2.0
    min_value = 1.0
    v2 = limit_value(v, max_value, min_value)
    assert v2 <= max_value
    assert v2 >= min_value
    v = np.random.rand(3)
    max_value = 0.3
    min_value = 0.1
    v2 = limit_value(v, max_value, min_value)
    assert (v2 <= max_value).all()
    assert (v2 >= min_value).all()
    v = np.random.rand(3)
    max_value = np.array([0.8, 0.5, 0.15])
    min_value = np.repeat(0.1, 3)
    v2 = limit_value(v, max_value, min_value)
    assert (v2 <= max_value).all()
    assert (v2 >= min_value).all()


def test_check_limits():
    v = np.random.rand()
    max_value = 1.0
    min_value = 0.0
    r = check_limits(v, max_value, min_value)
    assert r is True
    v = 1.1
    r = check_limits(v, max_value, min_value)
    assert r is False
    v = -0.1
    r = check_limits(v, max_value, min_value)
    assert r is False
    v = np.random.rand(3)
    r = check_limits(v, max_value, min_value)
    v[0] = 1.1
    r = check_limits(v, max_value, min_value)
    assert r is False
    v = np.array([0.5, 0.4, 0.3])
    max_value = np.array([0.8, 0.5, 0.45])
    min_value = np.repeat(0.1, 3)
    r = check_limits(v, max_value, min_value)
    assert r is True
    v[0] = 0.9
    r = check_limits(v, max_value, min_value)
    assert r is False
