import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from kmn.tf_util import apply_prediction
from kmn.util import invert_trans
from kmn.kmn_model import KmnModel, KmnMultiModel


class KmnModelConf(KmnModel):
    """
    Implementation of kinematic morphing networks that predicts transformation and configuration parameters
    """
    def train(self, x_train, y_train, x_test=None, y_test=None, i_pred=0):
        """
        Trains the neural network parameters
        :param x_train: depth image inputs for training [num_data x (width*height)]
        :param y_train: parameter outputs for training [num_data x num_param]
        :param x_test: depth image inputs for testing [num_data x (width*height)]
        :param y_test: parameter outputs for training [num_data x num_param]
        :param i_pred: integer that specifies the prediction model
        """
        # Split training data in transformation and configuration part
        n_t = self.c['N_TRANS']
        y_train = [y_train[:, :n_t], y_train[:, n_t:]]
        if y_test is not None:
            y_test = [y_test[:, :n_t], y_test[:, n_t:]]

        # Call regular training function
        KmnModel.train(self, x_train, y_train, x_test, y_test, i_pred)

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
            param_hist[0, :] = np.zeros(self.c['OUT_DIM'])
            y_hist[0, :] = np.zeros(self.c['OUT_DIM'])

        for i in range(n_pred):
            with tf.device('/cpu:0'):
                trans_param, conf_param = self.pred(D.reshape(1, -1), i_pred=i)
            if i > 0:
                trans_param = trans_param * self.c['DELTA']
            T = self.scene.param_to_trans(trans_param)
            T_inv = invert_trans(T)
            D, P = apply_prediction(self.c, P, T_inv)
            T_pred = np.matmul(T_pred, T)  # concatenate transformations
            if return_hist:
                D_hist[i+1, :] = D.reshape(1, -1)
                param_hist[i+1, :] = np.hstack([trans_param, conf_param])
                y_hist[i+1, :] = np.hstack([self.scene.trans_to_param(T_pred).reshape(1, -1), conf_param])

        trans_pred = self.scene.trans_to_param(T_pred)  # convert transformation to parameter
        conf_pred = np.squeeze(conf_param)
        param_pred = {"trans": trans_pred, "conf": conf_pred}

        if return_hist:
            return param_pred, D, P, D_hist, param_hist, y_hist
        else:
            return param_pred, D, P

    def augment_datapoint(self, i, x, y, n_pred):
        """
        Applies the current model to augment the dataset (x, y)
        :param x: depth image inputs [num_data x (width*height)]
        :param y: parameter outputs [num_data x num_param]
        :param n_pred: number of network predictions

        :return x_aug: augmented depth image [1 x width*height]
        :return y_aug: augmented parameter [num_param]
        """
        x_aug, y_aug = KmnModel.augment_datapoint(self, i, x, y, n_pred)
        y_aug = np.hstack([y_aug, y[self.c['N_TRANS']:]])  # append configuration parameter to y
        return x_aug, y_aug


class KmnMultiModelConf(KmnModelConf, KmnMultiModel):
    """
    Extension of KMN that uses at each prediction step a different model
    """
