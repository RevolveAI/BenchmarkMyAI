
import tensorflow as tf
from rbm.utils import plugins
from rbm.backends.vision import ImageProcessing
from rbm.backends import TensorflowBackend


@plugins.register
class KerasModels(TensorflowBackend, ImageProcessing):
    variants = ['DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0',
                'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
                'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'InceptionResNetV2',
                'InceptionV3', 'MobileNet', 'MobileNetV2', 'MobileNetV3Large', 'MobileNetV3Small',
                'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2',
                'ResNet50', 'ResNet50V2', 'VGG16', 'VGG19', 'Xception']

    def __init__(self, model_name, device=None, img_size=(224, 224), batch_size=1, **kwargs):
        TensorflowBackend.__init__(self, device=device)
        ImageProcessing.__init__(self, img_size=img_size, batch_size=batch_size)
        self.model_name = model_name
        self.__name__ = self.model_name
        self.kwargs = kwargs
        self._model = None

    def __call__(self):
        model = eval('tf.keras.applications.' + self.model_name)
        include_top = self.kwargs.pop('include_top', False)
        weights = self.kwargs.pop('weights', None)
        model = model(input_shape=(*self.img_size, 3), include_top=include_top, weights=weights, **self.kwargs)
        self._model = model

    def predict(self, inputs):
        return self._model.predict(inputs)

