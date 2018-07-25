import pytest
import numpy as np
from importlib.machinery import SourceFileLoader
from PIL import Image
from kmn.dataset_generator import DatasetGenerator
from kmn.tf_util import read_cfg, apply_prediction
from kmn.util import invert_trans, error_mae
from kmn import render_depth_image
import matplotlib.pyplot as plt
import os
from definitions import ROOT_DIR
os.chdir('../')

VIS_TEST = False


@pytest.fixture(params=["box", "door", "box_trans", "box_complex"])
def cfg(request):
    scene_name = request.param
    c = read_cfg(ROOT_DIR + "/kmn/scenes/" + scene_name + "/cfgs/basic", "tmp/")
    print('data', c['DATA_DIR'])
    return c


@pytest.fixture
def scene(cfg):
    s = SourceFileLoader("", ROOT_DIR + '/kmn/scenes/' + cfg['SCENE'] + "/scene.py").load_module()
    return s


def test_param_transformation(scene, cfg):
    trans_param = np.random.rand(cfg['N_TRANS'])
    T = scene.param_to_trans(trans_param)
    trans_param_2 = scene.trans_to_param(T)
    assert np.isclose(trans_param, trans_param_2).all()


@pytest.fixture
def model(cfg):
    if cfg['N_CONF'] > 0:
        if cfg['MULTI_MODEL']:
            from kmn.kmn_model_conf import KmnMultiModelConf
            m = KmnMultiModelConf(cfg)
        else:
            from kmn.kmn_model_conf import KmnModelConf
            m = KmnModelConf(cfg)
    else:
        if cfg['MULTI_MODEL']:
            from kmn.kmn_model import KmnMultiModel
            m = KmnMultiModel(cfg)
        else:
            from kmn.kmn_model import KmnModel
            m = KmnModel(cfg)
    return m


@pytest.fixture
def data_gen(cfg):
    dg = DatasetGenerator(cfg)
    return dg


def test_pred(model, data_gen, cfg):
    [x_train, y_train, x_test, y_test] = data_gen.load_dataset(small=True)
    n = 1
    D = x_test[0:n, :]
    param = model.pred(D)
    if cfg['N_CONF'] is 0:
        assert n == param.shape[0]
        assert cfg['N_TRANS'] == param.shape[1]
    else:
        assert n == param[0].shape[0]
        assert cfg['N_TRANS'] == param[0].shape[1]
        assert cfg['N_CONF'] == param[1].shape[1]


def test_multi_step_pred(model, data_gen, cfg):
    [x_train, y_train, x_test, y_test] = data_gen.load_dataset(small=True)
    D = x_train[0:1, :]
    P = np.load(cfg['DATA_DIR'] + 'raw/0_pts.npy')
    n_pred = 3
    trans, D, P = model.multi_step_pred(D, P, n_pred)
    assert True


def test_ground_truth_pred(cfg, data_gen, scene):
    [x_train, y_train, x_test, y_test] = data_gen.load_dataset(small=True)
    P0 = np.load(cfg['DATA_DIR'] + 'raw/0_pts.npy')
    D0 = Image.open(cfg['DATA_DIR'] + 'raw/0_d.ppm')
    D = Image.open(cfg['DATA_DIR'] + 'raw/1_d.ppm')
    P = np.load(cfg['DATA_DIR'] + 'raw/1_pts.npy')
    y = np.loadtxt(cfg['DATA_DIR'] + 'raw/1_param.dat')

    T = scene.param_to_trans(y)
    print(y)

    D2, P2 = apply_prediction(cfg, P0, T)

    print(np.max(P, axis=0))
    print(np.max(P2, axis=0))
    print(np.min(P, axis=0))
    print(np.min(P2, axis=0))

    plt.figure(1)
    plt.imshow(D)
    plt.figure(2)
    plt.imshow(D2)

    if VIS_TEST:
        plt.show()
    assert True


def test_augment_datapoint(model, data_gen, cfg):
    [x_train, y_train, x_test, y_test] = data_gen.load_dataset(small=True)
    x_train, y_train = x_train[0:2, :], y_train[0:2, :]
    x_aug, y_aug = model.augment_dataset(x_train, y_train, n_pred=3)
    assert np.isclose(y_aug[:, cfg['N_TRANS']:], y_train[:, cfg['N_TRANS']:]).all()
    assert x_train.shape == x_aug.shape
    assert y_train.shape == y_aug.shape


def test_render_depth_image(cfg, data_gen):
    [x_train, y_train, x_test, y_test] = data_gen.load_dataset(small=True)
    P = np.load(cfg['DATA_DIR'] + 'raw/0_pts.npy')
    D = Image.open(cfg['DATA_DIR'] + 'raw/0_d.ppm')
    cam_param = np.loadtxt(cfg['DATA_DIR'] + 'raw/0_camParam.dat')
    T_world_cam = np.loadtxt(cfg['DATA_DIR'] + 'raw/0_camT.dat')
    T_cam_world = invert_trans(T_world_cam)
    D = np.array(D)
    D2 = render_depth_image.f(P, T_cam_world, cfg['CAM_PARAM'])
    plt.figure(1)
    plt.imshow(D)
    plt.figure(2)
    plt.imshow(D2)
    plt.figure(3)
    plt.imshow(D2.astype(np.double) - D.astype(np.double))
    mae = error_mae(D2.astype(np.double), D.astype(np.double))
    if VIS_TEST:
        plt.show()
    assert mae < 5e-1
