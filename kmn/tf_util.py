import os
import numpy as np
import configparser
from kmn.util import resize_image, transform_pointcloud, invert_trans, create_folder
from kmn import render_depth_image
import socket
from datetime import datetime
from shutil import copyfile
from definitions import ROOT_DIR


def read_cfg(cfg_name, log_name):
    """
    :param cfg_name: path to a .cfg file
    :param log_name: path to a directory where to save logging files
    :return c: dictionary
    """
    cfg = configparser.ConfigParser()
    cfg.read(cfg_name + ".cfg")

    c = dict()
    c['SEED'] = cfg.getint('general', 'SEED')                        # random seed
    c['SCENE'] = cfg.get('general', 'SCENE')                         # scene name (e.g., 'box', 'door')
    c['N_DATA'] = cfg.getint('general', 'N_DATA')                    # amount of training datapoints
    c['N_DATA_EVAL'] = cfg.getint('general', 'N_DATA_EVAL')          # amount of evaluation datapoints
    c['N_DATA_AUG'] = cfg.getint('general', 'N_DATA_AUG')            # amount of augmentation datapoints
    c['DATASET'] = cfg.get('general', 'DATASET')                     # dataset name
    c['CREATE_DATA'] = cfg.getboolean('general', 'CREATE_DATA')      # if dataset should be created from raw data
    c['CPU_ONLY'] = cfg.getboolean('general', 'CPU_ONLY')            # training only on cpu
    c['DEBUG'] = cfg.getboolean('general', 'DEBUG')                  # sets certain parameters for testing (see below)
    c['DESCR'] = cfg.get('general', 'DESCR', fallback='')            # description of the cfg file

    c['MODEL'] = cfg.get('model', 'MODEL')                                     # specifies the model to use
    c['WIDTH'] = cfg.getint('model', 'WIDTH')                                  # width of images
    c['HEIGHT'] = cfg.getint('model', 'HEIGHT')                                # height of images
    c['DEPTH'] = cfg.getint('model', 'DEPTH')                                  # depth of images
    c['OUT_DIM'] = cfg.getint('model', 'OUT_DIM')                              # total number of parameters
    c['N_CONF'] = cfg.getint('model', 'N_CONF', fallback=0)                    # amount of configuration parameters
    c['N_TRANS'] = c['OUT_DIM'] - c['N_CONF']                                  # amount of transformation parameters
    c['N_PRED'] = cfg.getint('model', 'N_PRED')                                # amount of iterative network predictions
    c['MULTI_MODEL'] = cfg.getboolean('model', 'MULTI_MODEL', fallback=False)  # use different model for each prediction
    c['BATCH'] = cfg.getint('training', 'BATCH')                               # batch size
    c['N_EPOCH'] = cfg.getint('training', 'N_EPOCH')                           # number of training epochs per iteration
    c['DELTA'] = cfg.getfloat('model', 'DELTA', fallback=1.0)                  # factor for partial parameter updates

    # specifies the number of conv channels for all layers
    try:
        c['N_CONV'] = np.fromstring(cfg.get('model', 'N_CONV'), dtype=int, sep=' ')
    except configparser.NoOptionError:
        c['N_CONV'] = np.array([2, 4, 8, 16, 32])

    if c['DEBUG']:
        c['N_EPOCH'] = 12
        c['N_DATA_EVAL'] = 10
        c['N_DATA_AUG'] = 5

    # definition of log and result directories
    c['LOG_PREFIX'] = "0/"
    c['LOG_NAME'] = log_name
    c['LOG_DIR'] = ROOT_DIR + "/kmn/scenes/" + c['SCENE'] + "/logs/" + c['LOG_NAME'] + "/"
    c['DATA_DIR'] = ROOT_DIR + "/kmn/scenes/" + c['SCENE'] + "/data/"
    c['RESULT_DIR'] = ROOT_DIR + "/kmn/scenes/" + c['SCENE'] + "/results/" + c['LOG_NAME'] + "/"
    c['PARAM_NAMES'] = np.genfromtxt(c['DATA_DIR'] + 'raw/paramNames.dat', dtype=np.str, delimiter='\n')

    # camera parameters for the transformation of point clouds from world to camera frame
    c['CAM_PARAM'] = np.loadtxt(c['DATA_DIR'] + 'raw/0_camParam.dat')
    c['T_WORLD_CAM'] = np.loadtxt(c['DATA_DIR'] + 'raw/0_camT.dat')
    c['T_CAM_WORLD'] = invert_trans(c['T_WORLD_CAM'])

    return c


def init_cfg(cfg_name, date_str=None, pc_str=None):
    """
    Initializes the training by creating a folder structure for results and logging
    :param cfg_name: path to a .cfg file
    :param date_str: date when training is performed
    :param pc_str: name of the machine on which it is trained
    :return c: dictionary
    """
    if pc_str is None:
        log_name = os.path.basename(os.path.realpath(cfg_name + ".cfg"))[:-4] + "_" + socket.gethostname()[:5]
    else:
        log_name = os.path.basename(os.path.realpath(cfg_name + ".cfg"))[:-4] + "_" + pc_str

    if date_str is None:
        time = datetime.now()
        log_name = log_name + "_" + time.strftime("%m%d_%H%M")
    else:
        log_name = log_name + "_" + date_str

    c = read_cfg(cfg_name, log_name)

    create_folder(c['LOG_DIR'])
    create_folder(c['RESULT_DIR'])
    np.save(c['LOG_DIR'] + "run.npy", c)
    copyfile(cfg_name + ".cfg", c['RESULT_DIR'] + "run.cfg")

    print('descr  :', c['DESCR'])
    print('log dir:', c['LOG_DIR'])
    print('scene  :', c['SCENE'])
    print('cfg    :', c['LOG_NAME'])
    print('data   :', c['DATASET'])
    print('model  :', c['MODEL'])
    print('param  :', c['PARAM_NAMES'])
    return c


def load_cfg(cfg_path):
    """
    loads an existing config from file
    :param cfg_path: path to a .npy file
    :return c: dictionary
    """
    c = np.load(cfg_path)[()]
    return c


def set_device(c):
    """
    returns a string for training on cpu or gpu
    """
    if c['CPU_ONLY']:
        return '/cpu:0'
    else:
        return '/device:GPU:1'


def apply_prediction(c, P, T):
    """
    applies an affine transformation on a point cloud and generates a depth image
    :param c: cfg dictionary
    :param P: point cloud
    :param T: transformation matrix

    :return D_pred: transformed depth image
    :return P_pred: transformed point cloud
    """
    P_pred = transform_pointcloud(T, P)  # apply transformation on point cloud
    D_pred = render_depth_image.f(P_pred, c['T_CAM_WORLD'], c['CAM_PARAM'])  # render a depth image
    D_pred = resize_image(D_pred, c['WIDTH'], c['HEIGHT'])
    return D_pred, P_pred
