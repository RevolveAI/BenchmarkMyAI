

#%%
import tensorflow as tf
from official.vision.beta.modeling.backbones.spinenet import SpineNet as spineNet
from ..utils import plugins


@plugins.register
class SpineNetBackbone:
    def __init__(self, img_size, batch_size=None):
        self.img_size = img_size
        self.__framework__ = 'TensorFlow ' + tf.__version__
        self.__name__ = 'spinenet-backbone'
        self._model = None

    def __call__(self):
        img_size = self.img_size
        assert img_size[0] == img_size[1], 'input must be a square image'
        assert img_size[0]/2**5 % 2 == 0, 'image size is not valid, try something else e.g. some valid values are (28,28), (128,128), (192,192)'
        target_size = img_size
        #%%
        outshape = (4,2)
        outsize = outshape[0]*outshape[1]
        inshape = (*target_size,3)
        #%%
        inputs = tf.keras.layers.Input(shape=inshape,dtype=tf.uint8)
        x = tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(
            tf.cast(data, tf.float32)))(inputs)
        backbone = spineNet(tf.keras.layers.InputSpec(shape=(None,*target_size,3)))
        xs = backbone(x)
        xs = [tf.keras.layers.GlobalAveragePooling2D()(output) for output in xs.values()]
        x = tf.keras.layers.Concatenate(axis = 1)(xs)
        x = tf.keras.layers.Dense(outsize)(x)
        output = tf.keras.layers.Reshape(outshape)(x)
        #%%
        model = tf.keras.Model(inputs=inputs, outputs=output)
        self._model = model

    def predict(self, inputs):
        return self._model.predict(inputs)