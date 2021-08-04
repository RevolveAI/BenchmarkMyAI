
import tensorflow as tf
from . import plugins


@plugins.register
class KerasModels:
    variants = ['DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0',
                'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
                'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'InceptionResNetV2',
                'InceptionV3', 'MobileNet', 'MobileNetV2', 'MobileNetV3Large', 'MobileNetV3Small',
                'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2',
                'ResNet50', 'ResNet50V2', 'VGG16', 'VGG19', 'Xception']

    def __init__(self, model_name, img_size, batch_size=None, **kwargs):
        self.model_name = model_name
        self.img_size = img_size
        self.kwargs = kwargs
        self.__framework__ = 'TensorFlow ' + tf.__version__
        self._model = None

    def __call__(self):
        model = eval('tf.keras.applications.' + self.model_name)
        include_top = self.kwargs.pop('include_top', False)
        model = model(input_shape=(*self.img_size, 3), include_top=include_top, **self.kwargs)
        self.__name__ = model.get_config()['name']
        self._model = model

    def predict(self, inputs):
        return self._model.predict(inputs)

