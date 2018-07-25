from keras.models import Model
from keras.layers import Input, Reshape, Dense, MaxPooling2D, Conv2D, Lambda
import numpy as np


def scaling(x):
    """
    Rescale depth values from range [0-255] to range [1-0]
    """
    return -x/255.0 + 1.0


def scaling_shape(input_shape):
    return input_shape


def create(c):
    """
    @type c: dict

    :param c: config dictionary
    :return model: keras model
    """

    n_h = c['HEIGHT']
    n_w = c['WIDTH']

    Scaling = Lambda(scaling, output_shape=scaling_shape, name='Scaling')

    I = Input(shape=(n_h * n_w,), dtype='float32', name='I')
    Im = Reshape((n_h, n_w, 1), input_shape=(n_h * n_w,), name='Im')(I)

    Im = Scaling(Im)

    n_conv1 = c['N_CONV'][0]
    n_conv2 = c['N_CONV'][1]
    n_conv3 = c['N_CONV'][2]
    n_conv4 = c['N_CONV'][3]
    n_conv5 = c['N_CONV'][4]

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

    model.compile(loss='mean_squared_error', metrics=['mean_absolute_error'], optimizer='adam')

    model.summary()

    return model
