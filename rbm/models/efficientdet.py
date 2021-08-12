

import os
import wget
import tarfile
import subprocess
import tensorflow as tf
import shutil
from ..utils import plugins
from rbm.backends.vision import ImageProcessing
from rbm.backends import TensorflowBackend


@plugins.register
class EfficientDet(TensorflowBackend, ImageProcessing):
    variants = ['efficientdet-d0', 'efficientdet-d1', 'efficientdet-d2',
                'efficientdet-d3', 'efficientdet-d4', 'efficientdet-d5',
                'efficientdet-d6', 'efficientdet-d7']

    def __init__(self, model_name, device, batch_size=1):
        TensorflowBackend.__init__(self, device=device)
        ImageProcessing.__init__(self, img_size=(224, 224), batch_size=batch_size)
        self.model_name = model_name
        self.export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', 'efficientdet', 'tmp')
        self.export_model_dir = os.path.join(self.export_dir, 'model')
        self._model_ = None
        self.__name__ = model_name

    def download_checkpoints(self):
        try:
            shutil.rmtree(self.export_dir)
        except:
            pass
        os.makedirs(self.export_dir)
        filename = wget.download(f'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/{self.model_name}.tar.gz',
                               out=self.export_dir)
        tar = tarfile.open(filename, "r:gz")
        tar.extractall(self.export_dir)
        tar.close()
        os.remove(filename)
        ckpt_path = os.path.join(self.export_dir, self.model_name)
        self.ckpt_path = ckpt_path

    def export_model(self):
        os.makedirs(self.export_model_dir)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', 'efficientdet')
        # %%
        process = subprocess.Popen(['python', os.path.join(model_path, 'model_inspect.py'), '--runmode', 'saved_model',
                                    '--model_name', self.model_name, '--ckpt_path', self.ckpt_path,
                                    '--saved_model_dir', self.export_model_dir,
                                    '--batch_size', str(self.batch_size),
                                    '--hparams', f"mixed_precision=True"],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        # print(stderr.decode("utf-8"))
        return [True]

    def load_model(self):
        model = tf.saved_model.load(self.export_model_dir)
        model = model.signatures['serving_default']
        self._model_ = model

    def preprocess(self, input_images):
        images = input_images.copy()
        phi = int(self.model_name.split('-d')[1])
        res = 512 + phi * 128
        images = tf.cast(images, tf.float32)
        images = tf.image.resize_with_pad(images, res, res)
        if len(images.shape) == 3:
            images = tf.expand_dims(images, 0)
            batches = 1
        else:
            batches = images.shape[0]
        images.set_shape((batches, res, res, 3))
        return tf.cast(images, tf.uint8)

    def delete_tmp_dir(self):
        shutil.rmtree(self.export_dir)

    def __call__(self):
        self.download_checkpoints()
        _ = self.export_model()
        assert _[0], _[1]
        self.load_model()
        self.delete_tmp_dir()

    def predict(self, input_images):
        return self._model_(input_images)



