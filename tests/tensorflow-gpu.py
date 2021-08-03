import tensorflow as tf
from absl import logging
tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
# logging.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU'))<1:
    raise RuntimeError('GPU not correctly configured')
