import tensorflow as tf
from tensorflow.python.client import device_lib
# from keras.backend.tensorflow_backend import set_session

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, UpSampling2D, \
Convolution2D, Reshape, Dense, MaxPooling2D, concatenate, Add

from utils.loss_functions import *


# U-Net model Basic
def unetModel_basic_4(input_height, input_width, nChannels, lr_rate=1e-3, dropout_ratio=0.2, activation='relu'):
    inputs = Input(shape=(input_height, input_width, nChannels))

    conv1 = Convolution2D(16, (3, 3), activation=activation, padding='same')(inputs)
    conv1 = Dropout(dropout_ratio)(conv1)
    conv1 = Convolution2D(16, (3, 3), activation=activation, padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(32, (3, 3), activation=activation, padding='same')(pool1)
    conv2 = Dropout(dropout_ratio)(conv2)
    conv2 = Convolution2D(32, (3, 3), activation=activation, padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(64, (3, 3), activation=activation, padding='same')(pool2)
    conv3 = Dropout(dropout_ratio)(conv3)
    conv3 = Convolution2D(64, (3, 3), activation=activation, padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(128, (3, 3), activation=activation, padding='same')(pool3)
    conv4 = Dropout(dropout_ratio)(conv4)
    conv4 = Convolution2D(128, (3, 3), activation=activation, padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(256, (3, 3), activation=activation, padding='same')(pool4)
    conv5 = Dropout(dropout_ratio)(conv5)
    conv5 = Convolution2D(256, (3, 3), activation=activation, padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Convolution2D(512, (3, 3), activation=activation, padding='same')(pool5)
    conv6 = Dropout(dropout_ratio)(conv6)
    conv6 = Convolution2D(512, (3, 3), activation=activation, padding='same')(conv6)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv5], axis=3)
    conv7 = Convolution2D(256, (3, 3), activation=activation, padding='same')(up1)
    conv7 = Dropout(dropout_ratio)(conv7)
    conv7 = Convolution2D(256, (3, 3), activation=activation, padding='same')(conv7)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv4], axis=3)
    conv8 = Convolution2D(128, (3, 3), activation=activation, padding='same')(up2)
    conv8 = Dropout(dropout_ratio)(conv8)
    conv8 = Convolution2D(128, (3, 3), activation=activation, padding='same')(conv8)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=3)
    conv9 = Convolution2D(64, (3, 3), activation=activation, padding='same')(up3)
    conv9 = Dropout(dropout_ratio)(conv9)
    conv9 = Convolution2D(64, (3, 3), activation=activation, padding='same')(conv9)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=3)
    conv10 = Convolution2D(32, (3, 3), activation=activation, padding='same')(up4)
    conv10 = Dropout(dropout_ratio)(conv10)
    conv10 = Convolution2D(32, (3, 3), activation=activation, padding='same')(conv10)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=3)
    conv11 = Convolution2D(16, (3, 3), activation=activation, padding='same')(up5)
    conv11 = Dropout(dropout_ratio)(conv11)
    conv11 = Convolution2D(16, (3, 3), activation=activation, padding='same')(conv11)

    conv12 = Convolution2D(1, (1, 1), activation='sigmoid', name='main_output')(conv11)

    conv12 = Reshape((input_height * input_width, 1))(conv12)

    model = Model(inputs=inputs, outputs=conv12)

    optAdam = Adam(lr=lr_rate)
    model.compile(loss=dice_coef_loss, optimizer=optAdam, metrics=[dice_coef])#, sample_weight_mode="temporal")

    return model

# Res-U-Net model Basic
def unetModel_residual(input_height, input_width, nChannels, lr_rate=1e-3, dropout_ratio=0.2, activation='relu'):
    inputs = Input(shape=(input_height, input_width, nChannels))
    num_features = 16

    conv1 = Convolution2D(num_features * pow(2, 0), (3, 3), activation=activation, padding='same')(inputs)
    conv1 = Dropout(dropout_ratio)(conv1)
    conv1 = Convolution2D(num_features * pow(2, 0), (3, 3), activation=activation, padding='same')(conv1)
    # Residual Conneciton
    shortcut = Convolution2D(num_features, kernel_size=(1, 1))(inputs)
    output = Add()([conv1, shortcut])
    pool1 = MaxPooling2D(pool_size=(2, 2))(output)

    conv2 = Convolution2D(num_features * pow(2, 1), (3, 3), activation=activation, padding='same')(pool1)
    conv2 = Dropout(dropout_ratio)(conv2)
    conv2 = Convolution2D(num_features * pow(2, 1), (3, 3), activation=activation, padding='same')(conv2)
    # Residual Conneciton
    shortcut = Convolution2D(num_features * pow(2, 1), kernel_size=(1, 1))(pool1)
    output = Add()([conv2, shortcut])
    pool2 = MaxPooling2D(pool_size=(2, 2))(output)

    conv3 = Convolution2D(num_features * pow(2, 2), (3, 3), activation=activation, padding='same')(pool2)
    conv3 = Dropout(dropout_ratio)(conv3)
    conv3 = Convolution2D(num_features * pow(2, 2), (3, 3), activation=activation, padding='same')(conv3)
    # Residual Conneciton
    shortcut = Convolution2D(num_features * pow(2, 2), kernel_size=(1, 1))(pool2)
    output = Add()([conv3, shortcut])
    pool3 = MaxPooling2D(pool_size=(2, 2))(output)

    conv4 = Convolution2D(num_features * pow(2, 3), (3, 3), activation=activation, padding='same')(pool3)
    conv4 = Dropout(dropout_ratio)(conv4)
    conv4 = Convolution2D(num_features * pow(2, 3), (3, 3), activation=activation, padding='same')(conv4)
    # Residual Conneciton
    shortcut = Convolution2D(num_features * pow(2, 3), kernel_size=(1, 1))(pool3)
    output = Add()([conv4, shortcut])
    pool4 = MaxPooling2D(pool_size=(2, 2))(output)

    conv5 = Convolution2D(num_features * pow(2, 4), (3, 3), activation=activation, padding='same')(pool4)
    conv5 = Dropout(dropout_ratio)(conv5)
    conv5 = Convolution2D(num_features * pow(2, 4), (3, 3), activation=activation, padding='same')(conv5)
    # Residual Conneciton
    shortcut = Convolution2D(num_features * pow(2, 4), kernel_size=(1, 1))(pool4)
    output = Add()([conv5, shortcut])
    pool5 = MaxPooling2D(pool_size=(2, 2))(output)

    conv6 = Convolution2D(num_features * pow(2, 5), (3, 3), activation=activation, padding='same')(pool5)
    conv6 = Dropout(dropout_ratio)(conv6)
    conv6 = Convolution2D(num_features * pow(2, 5), (3, 3), activation=activation, padding='same')(conv6)
    # Residual Conneciton
    shortcut = Convolution2D(num_features * pow(2, 5), kernel_size=(1, 1))(pool5)
    output = Add()([conv6, shortcut])
    # pool6 = MaxPooling2D(pool_size=(2, 2))(output)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv5], axis=3)
    conv7 = Convolution2D(num_features * pow(2, 4), (3, 3), activation=activation, padding='same')(up1)
    conv7 = Dropout(dropout_ratio)(conv7)
    conv7 = Convolution2D(num_features * pow(2, 4), (3, 3), activation=activation, padding='same')(conv7)
    # Residual Conneciton
    shortcut = Convolution2D(num_features * pow(2, 4), kernel_size=(1, 1))(up1)
    conv7 = Add()([conv7, shortcut])

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv4], axis=3)
    conv8 = Convolution2D(num_features * pow(2, 3), (3, 3), activation=activation, padding='same')(up2)
    conv8 = Dropout(dropout_ratio)(conv8)
    conv8 = Convolution2D(num_features * pow(2, 3), (3, 3), activation=activation, padding='same')(conv8)
    # Residual Conneciton
    shortcut = Convolution2D(num_features * pow(2, 3), kernel_size=(1, 1))(up2)
    conv8 = Add()([conv8, shortcut])

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=3)
    conv9 = Convolution2D(num_features * pow(2, 2), (3, 3), activation=activation, padding='same')(up3)
    conv9 = Dropout(dropout_ratio)(conv9)
    conv9 = Convolution2D(num_features * pow(2, 2), (3, 3), activation=activation, padding='same')(conv9)
    # Residual Conneciton
    shortcut = Convolution2D(num_features * pow(2, 2), kernel_size=(1, 1))(up3)
    conv9 = Add()([conv9, shortcut])

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=3)
    conv10 = Convolution2D(num_features * pow(2, 1), (3, 3), activation=activation, padding='same')(up4)
    conv10 = Dropout(dropout_ratio)(conv10)
    conv10 = Convolution2D(num_features * pow(2, 1), (3, 3), activation=activation, padding='same')(conv10)
    # Residual Conneciton
    shortcut = Convolution2D(num_features * pow(2, 1), kernel_size=(1, 1))(up4)
    conv10 = Add()([conv10, shortcut])

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=3)
    conv11 = Convolution2D(num_features * pow(2, 0), (3, 3), activation=activation, padding='same')(up5)
    conv11 = Dropout(dropout_ratio)(conv11)
    conv11 = Convolution2D(num_features * pow(2, 0), (3, 3), activation=activation, padding='same')(conv11)
    # Residual Conneciton
    shortcut = Convolution2D(num_features * pow(2, 0), kernel_size=(1, 1))(up5)
    conv11 = Add()([conv11, shortcut])

    conv12 = Convolution2D(1, (1, 1), activation='sigmoid', name='main_output')(conv11)

    conv12 = Reshape((input_height * input_width, 1))(conv12)

    model = Model(inputs=inputs, outputs=conv12)

    optAdam = Adam(lr=lr_rate)
    model.compile(loss=dice_coef_loss, optimizer=optAdam, metrics=[dice_coef])#, sample_weight_mode="temporal")

    return model