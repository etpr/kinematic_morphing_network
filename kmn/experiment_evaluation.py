import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
from texttable import Texttable
from kmn.util import get_col_idx
from kmn.tf_util import load_cfg


def evaluate_experiment(runs, scene, save_plots=False):
    """
    Evaluates a set of different runs for a scene and creates plots/statistics to compare their performance
    :param runs: list of the runs to evaluate
    :param scene: name of the scene
    :param save_plots: option to save plots to file
    """
    leg = []
    t = Texttable()
    t.set_cols_width([30, 7, 7, 7, 7, 7, 7])
    t.set_precision(5)
    t.add_row(['Name', 'Train \nmae (trans)', 'Test \nmae (trans)',
               'Train mae (conf)', 'Test \nmae (conf)', 'Train \nmspe', 'Test \nmspe'])

    for i in range(len(runs)):
        cfg = load_cfg("kmn/scenes/" + scene + "/logs/" + runs[i] + "/run.npy")

        n_conf = "N_CONF" in cfg and cfg['N_CONF'] is not 0
        iter_start = 1
        train_min_i_trans = np.empty(shape=0)
        test_min_i_trans = np.empty(shape=0)
        train_min_i_conf = np.empty(shape=0)
        test_min_i_conf = np.empty(shape=0)
        mspe_train = None
        mspe_test = None

        for n_p in range(cfg['N_PRED']):
            if os.path.exists(cfg['LOG_DIR'] + str(n_p) + "/acc_trans.npy"):
                plt.figure(4 + i)
                # evaluate mean absolute error of training
                mae_trans = np.load(cfg['LOG_DIR'] + str(n_p) + "/acc_trans.npy")
                val_mae_trans = np.load(cfg['LOG_DIR'] + str(n_p) + "/val_acc_trans.npy")
                train_min_i_trans = np.append(train_min_i_trans, np.min(mae_trans))
                test_min_i_trans = np.append(test_min_i_trans, np.min(val_mae_trans))
                iter_end = iter_start + mae_trans.shape[0]
                plt.plot(np.arange(iter_start, iter_end), mae_trans, '-' + get_col_idx(n_p))
                plt.plot(np.arange(iter_start, iter_end), val_mae_trans, '--' + get_col_idx(n_p))
                if n_conf:
                    mae_conf = np.load(cfg['LOG_DIR'] + str(n_p) + "/acc_conf.npy")
                    val_mae_conf = np.load(cfg['LOG_DIR'] + str(n_p) + "/val_acc_conf.npy")
                    train_min_i_conf = np.append(train_min_i_conf, np.min(mae_conf))
                    test_min_i_conf = np.append(test_min_i_conf, np.min(val_mae_conf))
                    plt.plot(np.arange(iter_start, iter_end), mae_conf, '--' + get_col_idx(n_p))
                    plt.plot(np.arange(iter_start, iter_end), val_mae_conf, 'o' + get_col_idx(n_p))
                iter_start = iter_end + 1

            if os.path.exists(cfg['LOG_DIR'] + str(n_p) + "/Train_multi_step_error.npy"):
                mspe_train = np.load(cfg['LOG_DIR'] + str(n_p) + "/Train_multi_step_error.npy")
            if os.path.exists(cfg['LOG_DIR'] + str(n_p) + "/Test_multi_step_error.npy"):
                mspe_test = np.load(cfg['LOG_DIR'] + str(n_p) + "/Test_multi_step_error.npy")

        if n_conf:
            plt.legend(['trans (train)', 'trans (test)', 'conf (train)', 'conf (test)'])
        else:
            plt.legend(['trans (train)', 'trans (test)'])

        plt.xlabel('Training epochs')
        plt.ylabel('Mean absolute error')
        plt.yscale('log')
        plt.grid('on')
        plt.title(runs[i].replace('_', ' '))

        if save_plots:
            tikz_save('tikz_graphics/' + runs[i] + '.tex', figureheight='\\figureheight', figurewidth='\\figurewidth', show_info = False)

        t_iter = range(1, train_min_i_trans.shape[0]+1)
        plt.figure(0)
        plt.plot(t_iter, train_min_i_trans, ".-" + get_col_idx(i), markersize=1)
        plt.plot(t_iter, test_min_i_trans, ".:" + get_col_idx(i), markersize=1)
        plt.yscale('log')
        leg.append(runs[i].replace('_', ' ') + " train")
        leg.append(runs[i].replace('_', ' ') + " test")
        plt.title("trans")
        plt.xlabel('Iterations')
        plt.ylabel('Mean absolute error')

        table_row = [runs[i], np.min(train_min_i_trans), np.min(test_min_i_trans), 0, 0, 0, 0]

        if n_conf:
            plt.figure(1)
            plt.plot(t_iter, train_min_i_conf, ".-" + get_col_idx(i), markersize=1)
            plt.plot(t_iter, test_min_i_conf, ":" + get_col_idx(i), markersize=1)
            plt.title("conf")
            plt.yscale('log')
            plt.xlabel('Training iterations')
            plt.ylabel('Mean absolute error')
            table_row[3] = np.min(train_min_i_conf)
            table_row[4] = np.min(test_min_i_conf)

        # evaluate multi step prediction error
        if mspe_train is not None:
            mspe = np.sum(np.abs(mspe_train), axis=2)
            plt.figure(2)
            e_mean = np.mean(mspe, axis=0)
            plt.plot(np.arange(0, mspe_train.shape[1]), e_mean, get_col_idx(i)+"-", markersize=1)
            plt.yscale('log')
            plt.xlabel('Network predictions')
            plt.ylabel('Multi-step prediction error')
            table_row[5] = np.min(e_mean)
        if mspe_test is not None:
            mspe = np.sum(np.abs(mspe_test), axis=2)
            e_mean = np.mean(mspe, axis=0)
            plt.plot(np.arange(0, mspe_train.shape[1]), e_mean, get_col_idx(i)+'--', markersize=1)
            plt.yscale('log')
            plt.xlabel('Network predictions')
            plt.ylabel('Multi-step prediction error')
            table_row[6] = np.min(e_mean)

        t.add_row(table_row)

    print(t.draw())
    plt.figure(0)
    plt.legend(leg)
    if save_plots:
        tikz_save('tikz_graphics/error_training.tex', figureheight='\\figureheight', figurewidth='\\figurewidth', show_info = False)

    plt.figure(2)
    plt.legend(leg)
    if save_plots:
        tikz_save('tikz_graphics/error_prediction.tex', figureheight='\\figureheight', figurewidth='\\figurewidth', show_info = False)

    plt.show()