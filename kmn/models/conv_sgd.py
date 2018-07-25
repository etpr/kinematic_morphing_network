from keras.models import Model
from keras.layers import Input, Reshape, Dense, MaxPooling2D, Conv2D
from keras.optimizers import sgd
import numpy as np


def create(c):
    """
    @type c: dict

    :param c:config dictionary
    :return model: keras model
    """

    n_h = c['HEIGHT']
    n_w = c['WIDTH']

    I = Input(shape=(n_h * n_w,), dtype='float32', name='I')
    Im = Reshape((n_h, n_w, 1), input_shape=(n_h * n_w,), name='Im')(I)

    n_conv1 = 2
    n_conv2 = 4
    n_conv3 = 8
    n_conv4 = 16
    n_conv5 = 32

    x = Conv2D(n_conv1, (3, 3), activation='relu', padding='same', input_shape=(n_h, n_w, 1), name='conv1')(Im)
    x = MaxPooling2D()(x)
    x = Conv2D(n_conv2, (3, 3), activation='relu', padding='same', input_shape=(n_h / 2, n_w / 2, n_conv1),
               name='conv2')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(n_conv3, (3, 3), activation='relu', padding='same', input_shape=(n_h / 4, n_w / 4, n_conv2),
               name='conv3')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(n_conv4, (3, 3), activation='relu', padding='same', input_shape=(n_h / 8, n_w / 8, n_conv3),
               name='conv4')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(n_conv5, (3, 3), activation='relu', padding='same', input_shape=(n_h / 16, n_w / 16, n_conv4),
               name='conv5')(x)
    x = MaxPooling2D()(x)
    x = Reshape((np.int32(n_h / 32 * n_w / 32) * n_conv5,), input_shape=(n_h / 32, n_w / 32, n_conv5), name='x')(x)
    E = Dense(units=c['OUT_DIM'], activation='linear', name='E')(x)

    model = Model(inputs=[I], outputs=[E])

    model.compile(loss='mean_squared_error', metrics=['mean_absolute_error'], optimizer=sgd(lr=1e-3))

    model.summary()

    return model
