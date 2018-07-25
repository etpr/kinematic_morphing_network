import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from kmn.util import progress_bar, get_col_idx, create_folder


def evaluate_model(model, c, x_train, y_train, x_test, y_test, i_pred=0, show_plots=False):
    """
    Evaluates a KMN model and computes various statistics
    :param model: KmnModel
    :param c: cfg dictionary
    :param x_train: depth image inputs for training
    :param y_train: parameter outputs for training
    :param x_test: depth image inputs for testing
    :param y_test: parameter outputs for testing
    :param i_pred: amount of predictions for evaluation
    :param show_plots: option to visualize plots
    """
    create_folder(c['RESULT_DIR'] + c['LOG_PREFIX'])
    pdf = PdfPages(c['RESULT_DIR'] + c['LOG_PREFIX'] + 'result.pdf')

    if c['N_CONF'] is 0:
        p = ['trans']
    else:
        p = ['trans', 'conf']

    if c['MULTI_MODEL']:
        n_pred = i_pred + 1
    else:
        n_pred = c['N_PRED']

    for i in p:
        acc_hist = np.load(c['LOG_DIR'] + c['LOG_PREFIX'] + 'acc_' + i + '.npy')
        val_acc_hist = np.load(c['LOG_DIR'] + c['LOG_PREFIX'] + 'val_acc_' + i + '.npy')

        # plot train/test error over iterations
        plt.figure()

        plt.plot(np.log10(acc_hist), 'r')
        plt.plot(np.log10(val_acc_hist), 'g')
        plt.legend(['train', 'test'])
        plt.xlabel('Iterations')
        plt.ylabel('log MAE')
        txt = 'best test MAE: ' + "%.4f" % np.min(val_acc_hist) + ' after ' + str(np.argmin(val_acc_hist)) + \
              ' iterations\nbest train MAE: ' + "%.4f" % np.min(acc_hist) + ' after ' + str(np.argmin(acc_hist)) + \
              ' iterations'
        plt.figtext(0., 0., txt)
        plt.title(c['LOG_NAME'] + "; " + i + "; # param: " + str(model.m[i_pred].count_params()) + "; " +
                  c.get('DESCR', ''))
        pdf.savefig()
        plt.close()

    # iterate over datapoints and compute error stats
    for m in ['Train', 'Test']:
        print("\nEvaluation on " + m + " set")
        n_data = c['N_DATA_EVAL']
        error = np.zeros(shape=(n_data, n_pred + 1, c['OUT_DIM']))
        for i in range(n_data):
            progress_bar(i, n_data)
            if m == 'Train':
                x = x_train[i, :]
                y = y_train[i, :]
                p_id = i
            else:
                x = x_test[i, :]
                y = y_test[i, :]
                p_id = x_train.shape[0] + i

            P = np.load(model.folder_data_raw + str(p_id) + '_pts.npy')

            param, D, P, D_hist, param_hist, y_hist = \
                model.multi_step_pred(D=x, P=P, n_pred=n_pred, return_hist=True)

            # compute error
            error_i = y_hist - y
            error[i, :, :] = error_i

        # save error to file
        np.save(c['LOG_DIR'] + c['LOG_PREFIX'] + m + '_multi_step_error.npy', error)

        abs_error = np.abs(error)
        sum_error = np.sum(abs_error, axis=2)

        # visualize worst and best candidates
        e = sum_error[:, -1]
        n_ext = 5
        idx = np.arange(0, 5)
        idx_worst = np.argpartition(e, -n_ext)[-n_ext:]
        idx_best = np.argpartition(e, n_ext)[:n_ext]

        for (o, arg) in [["best", idx_best], ["worst", idx_worst], ["", idx]]:
            for i in arg:
                if m == 'Train':
                    x = x_train[i, :]
                    y = y_train[i, :]
                    p_id = i
                else:
                    x = x_test[i, :]
                    y = y_test[i, :]
                    p_id = x_train.shape[0] + i

                P = np.load(model.folder_data_raw + str(p_id) + '_pts.npy')

                param, D, P, D_hist, param_hist, y_hist = \
                    model.multi_step_pred(D=x, P=P, n_pred=n_pred, return_hist=True)

                # compute error
                error_i = y_hist - y

                pdf_pred = PdfPages(c['RESULT_DIR'] + '/' + c['LOG_PREFIX'] + m + "_" + o + str(i) + 'pred.pdf')
                for j in range(n_pred + 1):
                    plt.figure()
                    d = D_hist[j, :].reshape(c['HEIGHT'], c['WIDTH'])
                    plt.imshow(d, cmap=plt.gray())
                    # plt.imshow(d, cmap=plt.gray(), vmin=np.min(d), vmax=252)
                    plt.title(str(j) + " error = " + str(error_i[j]))
                    pdf_pred.savefig()
                    plt.close()
                plt.close()
                pdf_pred.close()

                np.savetxt(c['RESULT_DIR'] + '/' + c['LOG_PREFIX'] + m + "_" + o + str(i) + 'pred.dat',
                           np.vstack([y, y_hist]))

        # visualize error of predictions
        t = range(0, n_pred + 1)
        plt.figure()
        e_mean = np.mean(sum_error, axis=0)
        e_std = np.std(sum_error, axis=0)
        plt.errorbar(t, e_mean, 2 * e_std)
        plt.title(m + ' error of all outputs (' + str(n_data) + ' datapoints)')
        plt.xlabel('Number of predictions')
        plt.ylabel('MAE (mean + 2 std)')
        mIdx = np.argmin(e_mean)
        txt = 'Best prediction: ' + "%.4f" % (e_mean[mIdx]) + '+-' + "%.4f" % (2 * e_std[mIdx]) + ' after ' + \
              str(mIdx) + ' predictions'
        plt.figtext(0, 0, txt)
        plt.xticks(t)
        plt.grid()
        pdf.savefig()
        plt.close()

        for i in range(c['OUT_DIM']):
            i_error = abs_error[:, :, i]
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.hist(np.argmin(i_error, axis=1), normed=True, bins=range(0, n_pred + 2), align='left', rwidth=0.3)
            plt.ylabel('Best candidate rate')
            plt.ylim([0, 1])
            e_mean = np.mean(i_error, axis=0)
            e_std = np.std(i_error, axis=0)
            ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
            ax2.errorbar(t, e_mean, 2 * e_std, color='red')
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position('right')
            plt.title(m + ' error of output dimension ' + str(i) + ' (' + c['PARAM_NAMES'][i] + ')')
            plt.xlabel('Number of predictions')
            plt.ylabel('MAE (mean + 2 std)', color='red')
            mIdx = np.argmin(e_mean)
            txt = 'Best prediction: ' + "%.4f" % (e_mean[mIdx]) + '+-' + "%.4f" % (2 * e_std[mIdx]) + ' after ' + \
                  str(mIdx) + ' predictions'
            plt.figtext(0, 0, txt)
            plt.xticks(t)
            plt.grid()
            pdf.savefig()
            plt.close()

        # compute rate of improvement over prediction
        roi = np.empty((n_pred, c['OUT_DIM']))
        for i in range(n_pred):
            for j in range(c['OUT_DIM']):
                e = np.min(abs_error[:, 0:i + 1, j], axis=1)
                e_next = abs_error[:, i + 1, j]
                roi[i, j] = np.sum(np.less(e_next, e)) / n_data
        plt.figure()
        t = np.linspace(1, n_pred, n_pred)
        w = 0.1
        for i in range(c['OUT_DIM']):
            plt.bar(t + i * w, roi[:, i], width=w, color=get_col_idx(i))
        plt.legend(c['PARAM_NAMES'])
        plt.title('Rate of improvement over previous predictions (' + m + ')')
        plt.xlabel('Number of predictions')
        plt.ylabel('Improvement rate')
        plt.ylim([0.0, 1.0])
        plt.xticks(t)
        plt.grid()
        pdf.savefig()
        plt.close()

    pdf.close()
    if show_plots:
        plt.show()
