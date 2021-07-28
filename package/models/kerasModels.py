

import tensorflow as tf


class KerasModels:
    def __init__(self):
        pass

    def load(self, model, img_size):
        try:
            if model == 'ResNet50':
                assert img_size == (224, 224), 'Input image size must be (224,224)'
                model = tf.keras.applications.resnet50.ResNet50()
            elif model == 'ResNet101':
                assert img_size == (224, 224), 'Input image size must be (224,224)'
                model = tf.keras.applications.resnet.ResNet101()
            elif model == 'MobileNet':
                assert img_size == (224, 224), 'Input image size must be (224,224)'
                model = tf.keras.applications.mobilenet.MobileNet()
            elif model == 'MobileNetV2':
                assert img_size == (224, 224), 'Input image size must be (224,224)'
                model = tf.keras.applications.mobilenet_v2.MobileNetV2()
            elif 'MobileNetV3' in model:
                assert img_size == (224, 224), 'Input image size must be (224,224)'
                model = eval('tf.keras.applications.' + model + '(input_shape=(224, 224, 3))')
            elif 'EfficientNet' in model:
                model = eval('tf.keras.applications.efficientnet.' + model + '()')
            elif 'NASNet' in model:
                model = eval('tf.keras.applications.nasnet.' + model + '()')
            # try:
            #     model
            # finally:
            #     return [False, 'invalid model name']
        except AttributeError:
            return [False, 'invalid model name']
        return [True, model]

