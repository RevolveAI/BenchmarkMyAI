

#%%
import tensorflow as tf
from official.vision.beta.modeling.backbones.spinenet import SpineNet as spineNet
import cv2

def SpineNet(img_size, **kwargs):
    assert img_size == (28,28), 'image size must be (28,28)'
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
    model = tf.keras.Model(inputs= inputs,outputs=output)
    model.__framework__ = 'TensorFlow ' + tf.__version__ + ' | ' 'OpenCV ' + cv2.__version__
    model.__name__ = 'spinenet'
    return model