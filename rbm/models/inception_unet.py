

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D as Convolution2D
from tensorflow.keras.layers import Activation, BatchNormalization, MaxPooling2D, concatenate
from tensorflow.keras.layers import UpSampling2D
from ..utils import plugins


@plugins.register
class InceptionUNet:
    def __init__(self, img_size, batch_size=None, channels=3,  n_labels=2, numFilters=32, output_mode="sigmoid"):
        self.__framework__ = 'TensorFlow ' + tf.__version__
        self.__name__ = 'inception-unet'
        self.img_size = img_size
        # self.channels = channels
        self.n_labels = n_labels
        self.numFilters = numFilters
        self.output_mode = output_mode
        self._model = None

    def InceptionModule(self, inputs, numFilters=32):
        tower_0 = Convolution2D(numFilters, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)
        tower_0 = BatchNormalization()(tower_0)
        tower_0 = Activation("relu")(tower_0)

        tower_1 = Convolution2D(numFilters, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)
        tower_1 = BatchNormalization()(tower_1)
        tower_1 = Activation("relu")(tower_1)
        tower_1 = Convolution2D(numFilters, (3, 3), padding='same', kernel_initializer='he_normal')(tower_1)
        tower_1 = BatchNormalization()(tower_1)
        tower_1 = Activation("relu")(tower_1)

        tower_2 = Convolution2D(numFilters, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)
        tower_2 = BatchNormalization()(tower_2)
        tower_2 = Activation("relu")(tower_2)
        tower_2 = Convolution2D(numFilters, (3, 3), padding='same', kernel_initializer='he_normal')(tower_2)
        tower_2 = Convolution2D(numFilters, (3, 3), padding='same', kernel_initializer='he_normal')(tower_2)
        tower_2 = BatchNormalization()(tower_2)
        tower_2 = Activation("relu")(tower_2)

        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        tower_3 = Convolution2D(numFilters, (1, 1), padding='same', kernel_initializer='he_normal')(tower_3)
        tower_3 = BatchNormalization()(tower_3)
        tower_3 = Activation("relu")(tower_3)

        inception_module = concatenate([tower_0, tower_1, tower_2, tower_3], axis=3)
        return inception_module


    def __call__(self):
        input_shape = (*self.img_size, 3)
        inputs = Input(input_shape)

        conv1 = self.InceptionModule(inputs, self.numFilters)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.InceptionModule(pool1, 2 * self.numFilters)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.InceptionModule(pool2, 4 * self.numFilters)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.InceptionModule(pool3, 8 * self.numFilters)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = self.InceptionModule(pool4, 16 * self.numFilters)

        up6 = UpSampling2D(size=(2, 2))(conv5)
        up6 = self.InceptionModule(up6, 8 * self.numFilters)
        merge6 = concatenate([conv4, up6], axis=3)

        up7 = UpSampling2D(size=(2, 2))(merge6)
        up7 = self.InceptionModule(up7, 4 * self.numFilters)
        merge7 = concatenate([conv3, up7], axis=3)

        up8 = UpSampling2D(size=(2, 2))(merge7)
        up8 = self.InceptionModule(up8, 2 * self.numFilters)
        merge8 = concatenate([conv2, up8], axis=3)

        up9 = UpSampling2D(size=(2, 2))(merge8)
        up9 = self.InceptionModule(up9, self.numFilters)
        merge9 = concatenate([conv1, up9], axis=3)

        conv10 = Convolution2D(self.n_labels, (1, 1), padding='same', kernel_initializer='he_normal')(merge9)
        outputs = Activation(self.output_mode)(conv10)

        model = Model(inputs=inputs, outputs=outputs)

        self._model = model

    def predict(self, inputs):
        return self._model.predict(inputs)


