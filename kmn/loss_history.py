from keras.callbacks import Callback
from pickle import dump
from kmn.util import create_folder
import numpy as np


class LossHistory(Callback):
    """
    Keeps history of various losses/statistics during training
    """
    def __init__(self, log_dir, name, use_conf=False):
        self.log_dir = log_dir + name
        create_folder(self.log_dir)
        self.use_conf = use_conf

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.acc_trans = []
        self.val_acc_trans = []
        self.acc_conf = []
        self.val_acc_conf = []
        self.best_loss = np.inf
        self.weights = []
        self.i = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        val_loss = logs.get('val_loss')
        self.losses.append(logs.get('loss'))
        self.val_losses.append(val_loss)

        if self.use_conf:
            self.acc_trans.append(logs.get('Et_mean_absolute_error'))
            self.val_acc_trans.append(logs.get('val_Et_mean_absolute_error'))
            self.acc_conf.append(logs.get('Ec_mean_absolute_error'))
            self.val_acc_conf.append(logs.get('val_Ec_mean_absolute_error'))
        else:
            self.acc_trans.append(logs.get('mean_absolute_error'))
            self.val_acc_trans.append(logs.get('val_mean_absolute_error'))

        if val_loss < self.best_loss:
            dump(self.model.optimizer.get_config(), open(self.log_dir + 'best_opt_cfg.p', 'wb'))
            self.model.save_weights(self.log_dir + 'best_weights', overwrite=True)
            self.best_loss = val_loss
            self.model.save(self.log_dir + 'best_model.h5', overwrite=True)

        if (self.i % 10) == 0:
            np.save(self.log_dir + 'loss', self.losses)
            np.save(self.log_dir + 'val_loss', self.val_losses)
            np.save(self.log_dir + 'acc_trans', self.acc_trans)
            np.save(self.log_dir + 'val_acc_trans', self.val_acc_trans)
            if self.use_conf:
                np.save(self.log_dir + 'acc_conf', self.acc_conf)
                np.save(self.log_dir + 'val_acc_conf', self.val_acc_conf)

        self.i += 1
