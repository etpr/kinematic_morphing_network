import numpy as np
from PIL import Image
from multiprocessing import Pool
from kmn.util import progress_bar, create_folder


class DatasetGenerator:
    """
    Converts raw images to a (x,y) dataset in numpy format
    """
    def __init__(self, c):
        """
        :param c: cfg dictionary
        """
        self.c = c
        self.dim = self.c['WIDTH'] * self.c['HEIGHT']
        self.folder_data = c['DATA_DIR'] + "gen/" + self.c['DATASET']
        create_folder(self.folder_data)

    def load_datapoint(self, i):
        progress_bar(i, self.c['N_DATA'])

        filename = "../../" + "scenes/" + self.c['SCENE'] + "/data/raw/" + str(i)
        img = Image.open(filename + "_d.ppm")
        param = np.loadtxt(filename + "_param.dat")

        img = img.resize((self.c['WIDTH'], self.c['HEIGHT']), Image.ANTIALIAS)
        img = np.array(img)
        img = img.reshape(1, self.dim)
        return img, param

    def create_dataset(self):
        if not self.c['CREATE_DATA']:
            return

        p = Pool(20)
        data = p.map(self.load_datapoint, range(self.c['N_DATA']), chunksize=1)

        x = np.empty((self.c['N_DATA'], self.c['WIDTH'] * self.c['HEIGHT']), dtype=np.uint8)
        y = np.empty((self.c['N_DATA'], self.c['OUT_DIM']), dtype=np.float)

        for j in range(self.c['N_DATA']):
            x[j, :] = data[j][0]
            y[j, :] = data[j][1]

        print("x shape", x.shape)
        print("y shape", y.shape)
        print('mean', y.mean(axis=0))
        print('var', y.var(axis=0))
        print('max', y.max(axis=0))
        print('min', y.min(axis=0))

        # save dataset in numpy format
        np.save(self.folder_data + "/x", x)
        np.save(self.folder_data + "/y", y)

        # save a smaller dataset for testing/debugging purposes
        x_small = x[0:100, :]
        y_small = y[0:100, :]
        np.save(self.folder_data + "/x_small", x_small)
        np.save(self.folder_data + "/y_small", y_small)

    def load_dataset(self, small=False, train_test_ratio=0.8):
        if small:
            x = np.load(self.folder_data + "/x_small.npy")
            y = np.load(self.folder_data + "/y_small.npy")
        else:
            x = np.load(self.folder_data + "/x.npy")
            y = np.load(self.folder_data + "/y.npy")

        n = x.shape[0]
        idx = np.linspace(0, n - 1, n, dtype=int)
        idx_train = idx[0:int(train_test_ratio * n)]
        idx_test = idx[int(train_test_ratio * n):]
        x_train = x[idx_train, :]
        y_train = y[idx_train]
        x_test = x[idx_test, :]
        y_test = y[idx_test]

        print(x_train.shape[0], " data points for testing")
        print(x_test.shape[0], " data points for training")
        return [x_train, y_train, x_test, y_test]

