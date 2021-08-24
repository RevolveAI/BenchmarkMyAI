

import tensorflow as tf
from contextlib import contextmanager


class TensorflowBackend:

    def __init__(self, device=None, *args, **kwargs):
        if device is None:
            self.get_preferred_device()
        else:
            self.device = device
        self.__framework__ = 'TensorFlow ' + tf.__version__

    def get_preferred_device(self):
        if len(tf.config.list_physical_devices(device_type='GPU')) > 0:
            self.device = tf.config.list_physical_devices(device_type='GPU')[0].name.split('/physical_device:')[1]
        else:
            self.device = tf.config.list_physical_devices(device_type='CPU')[0].name.split('/physical_device:')[1]

    def memory_info(self):
        if 'GPU' in self.device:
            peak_memory_used = tf.config.experimental.get_memory_info(self.device)['peak']
            peak_memory_used = str(round(peak_memory_used * 1e-6, 2)) + 'MB'
        else:
            peak_memory_used = ''
        return peak_memory_used

    @contextmanager
    def device_placement(self):
        with tf.device(self.device):
            yield



