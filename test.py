import package as pk
#%%
import tensorflow as tf
#%%
model = tf.keras.applications.mobilenet.MobileNet(
    input_shape=None, alpha=1.0, depth_multiplier=1, dropout=0.001,
    include_top=True, weights='imagenet', input_tensor=None, pooling=None,
    classes=1000, classifier_activation='softmax'
)
#%%
benchmarker = pk.utils.benchmark(model=model, batch_size=256, img_size=(224,224), gpu_device=None)
#%%
benchmarks = benchmarker.execute()
