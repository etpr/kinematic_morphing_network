import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from importlib.machinery import SourceFileLoader
from keras.models import load_model
from keras.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf
from kmn.tf_util import set_device, apply_prediction
from kmn.loss_history import LossHistory
from kmn.util import invert_trans, progress_bar
from definitions import ROOT_DIR


class KmnModel:
    """
    Implementation of kinematic morphing networks that predicts transformation parameter
    """
    def __init__(self, c):
        """
        @type c: dict

        :param c: config dictionary
        """
        self.c = c
        self.scene = SourceFileLoader("", ROOT_DIR + '/kmn/scenes/' + self.c['SCENE'] + "/scene.py").load_module()
        self.tf_model = SourceFileLoader("", ROOT_DIR + '/kmn/models/' + self.c['MODEL'] + ".py").load_module()
        self.folder_data_raw = c['DATA_DIR'] + "raw/"
        self.m = [self.tf_model.create(self.c)]

    def load_model(self, file=None, i_pred=0):
        """
        :param file: keras model file
        :param i_pred: integer that specifies the prediction model
        """
        if file is None:
            self.m[i_pred] = load_model(self.c['LOG_DIR'] + self.c['LOG_PREFIX'] + 'best_model.h5')
        else:
            self.m[i_pred] = load_model(file)

    def train(self, x_train, y_train, x_test=None, y_test=None, i_pred=0):
        """
        Trains the neural network parameters
        :param x_train: depth image inputs for training [num_data x (width*height)]
        :param y_train: parameter outputs for training [num_data x num_param]
        :param x_test: depth image inputs for testing [num_data x (width*height)]
        :param y_test: parameter outputs for training [num_data x num_param]
        :param i_pred: integer that specifies the prediction model
        """
        with tf.device(set_device(self.c)):
            np.random.RandomState(self.c['SEED'])
            tf.set_random_seed(self.c['SEED'])
            log_cb = LossHistory(log_dir=self.c['LOG_DIR'], name=self.c['LOG_PREFIX'], use_conf=self.c['N_CONF'] > 0)
            conv_cb = EarlyStopping(monitor='val_loss', patience=100, mode="min")
            tboard_cb = TensorBoard(log_dir=self.c['LOG_DIR'])
            cb = [log_cb, conv_cb, tboard_cb]
            if x_test is None:
                self.m[i_pred].fit(x_train, y_train, epochs=self.c['N_EPOCH'], batch_size=self.c['BATCH'],
                                   callbacks=cb, shuffle=True)
            else:
                self.m[i_pred].fit(x_train, y_train, validation_data=(x_test, y_test), epochs=self.c['N_EPOCH'],
                                   batch_size=self.c['BATCH'], callbacks=cb, shuffle=True)

    def pred(self, D, i_pred=0):
        """
        :param D: depth images [num_data x (width*height)]

        :return param: predicted transformation parameters [num_data x num_param]
        """
        if D.ndim != 2:
            raise TypeError("Input depth images must be of shape [num_images x (width*height)]")
        param = self.m[0].predict(D)
        return param

    def multi_step_pred(self, D, P, n_pred=1, return_hist=False):
        """
        Applies the network multiple times on a data point
        :param D: depth image [width*height]
        :param P: pointcloud [(width*height) x 3]
        :param n_pred: number of network predictions
        :param return_hist: option that specifies if a history should be returned

        :return param_pred: predicted parameters
        :return D: predicted depth image
        :return P: predicted point cloud
        :return D_hist: history of depth images
        :return param_hist: history of parameter predictions
        :return y_hist: history of concatenated parameter predictions
        """
        T_pred = np.eye(4)

        if return_hist:
            D_hist = np.empty(shape=(n_pred + 1, self.c['WIDTH']*self.c['HEIGHT']))
            param_hist = np.empty(shape=(n_pred + 1, self.c['OUT_DIM']))
            y_hist = np.empty(shape=(n_pred + 1, self.c['OUT_DIM']))
            D_hist[0, :] = D.reshape(1, -1)
            param_hist[0, :] = self.scene.trans_to_param(T_pred)
            y_hist[0, :] = self.scene.trans_to_param(T_pred)

        for i in range(n_pred):
            with tf.device('/cpu:0'):
                param = self.pred(D.reshape(1, -1), i_pred=i)  # predict the parameter from depth image
            if i > 0:
                param = param * self.c['DELTA']
            T = self.scene.param_to_trans(param)
            T_inv = invert_trans(T)
            D, P = apply_prediction(self.c, P, T_inv)  # transform pointcloud and render a new depth image
            T_pred = np.matmul(T_pred, T)  # concatenate transformations
            if return_hist:  # book keeping
                D_hist[i+1, :] = D.reshape(1, -1)
                param_hist[i+1, :] = param.reshape(1, -1)
                y_hist[i+1, :] = self.scene.trans_to_param(T_pred)

        trans_pred = self.scene.trans_to_param(T_pred)  # convert transformation to parameter
        param_pred = {"trans": trans_pred}

        if return_hist:
            return param_pred, D, P, D_hist, param_hist, y_hist
        else:
            return param_pred, D, P

    def augment_dataset(self, x, y, n_pred):
        """
        Applies the current model to augment the dataset (x, y)
        :param x: depth image inputs [num_data x (width*height)]
        :param y: parameter outputs [num_data x num_param]
        :param n_pred: number of network predictions

        :return x_aug: augmented depth images [num_data x width*height]
        :return y_aug: augmented parameters [num_data x num_param]
        """
        print("Augmenting dataset")
        N = np.min([self.c['N_DATA_AUG'], x.shape[0]])

        x_aug = np.empty((N, x.shape[1]))
        y_aug = np.empty((N, y.shape[1]))

        for i in range(N):
            progress_bar(i, N)
            x_aug[i, :], y_aug[i, :] = self.augment_datapoint(i, x[i, :], y[i, :], n_pred)

        return [x_aug, y_aug]

    def augment_datapoint(self, i, x, y, n_pred):
        """
        Applies the model on a data point to generate an augmented data point
        :param i: data point id
        :param x: depth image input [width*height]
        :param y: parameter output [num_param]
        :param n_pred: number of network predictions

        :return x_aug: augmented depth image [1 x width*height]
        :return y_aug: augmented parameter [num_param]
        """
        P = np.load(self.folder_data_raw + str(i) + '_pts.npy')  # load corresponding point cloud from file
        param, D, P = self.multi_step_pred(D=x, P=P, n_pred=n_pred)
        T_pred = self.scene.param_to_trans(param['trans'])
        T_pred_inv = invert_trans(T_pred)
        T_y = self.scene.param_to_trans(y)
        T_aug = np.matmul(T_pred_inv, T_y)
        y_aug = self.scene.trans_to_param(T_aug)
        x_aug = np.reshape(D, (1, -1))
        return x_aug, y_aug


class KmnMultiModel(KmnModel):
    """
    Extension of KMN that uses at each prediction step a different model
    """
    def __init__(self, c):
        """
        @type c: dict

        :param c: config dictionary
        """
        KmnModel.__init__(self, c)
        # add a separate model for each prediction
        for i in range(self.c['N_PRED']-1):
            self.m.append(self.tf_model.create(self.c))

    def pred(self, D, i_pred=0):
        """
        :param D: depth images [num_data x (width*height)]
        :param i_pred: integer that specifies the prediction model

        :return param: predicted parameters [num_data x num_param]
        """
        if D.ndim != 2:
            raise TypeError("Input depth images must be of shape [num_data x (width*height)]")
        param = self.m[i_pred].predict(D)
        return param
