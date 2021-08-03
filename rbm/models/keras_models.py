

import tensorflow as tf


class KerasModels:
    def __init__(self):
        pass

    @staticmethod
    def load(model, img_size, **kwargs):
        model = eval('tf.keras.applications.' + model)
        include_top = kwargs.pop('include_top', False)
        model = model(input_shape=(*img_size, 3), include_top=include_top, **kwargs)
        model.__framework__ = 'TensorFlow ' + tf.__version__
        model.__name__ = model.get_config()['name']
        return model
