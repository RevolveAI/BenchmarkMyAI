# Deep Learning Models Benchmarks

A repository that contains code and benchmarks for commonly used models in RevolveAI.

## Table of Contents

* Directory Tree
* Install
* Run Benchmarks
* List of Available Models
* Run with Weights and Biases
* Add Custom Model



## Directory Tree

```
Repository
├── rbm
│   ├── __init__.py
│   ├── make
│   ├── models
│   │   ├── saved_models
│   │   ├── __init__.py
│   │   ├── keras_models.py
│   │   ├── efficientdet.py
│   │   ├── spinenet_backbone.py
│   │   └── inception_unet.py
│   ├── requirements.txt
│   └── utils
│       ├── __init__.py
│       ├── benchmark.py
│       ├── plugins.py
│       └── wandb.py
└── README.md
```

## Install

To use the package, first clone the the repository on your local machine and move to the location where it is cloned, then install all the requirements as following:

```
make
```

It will install all the required packages for this library.



## Run Benchmarks

Move to the directory where `rbm` folder is located. To calculate benchmarks, run the following code:

```python
# Import the library
import rbm
# Create benchmark instance
benchmarker = rbm.utils.Benchmark(model='ResNet50', batch_size=2, img_size=(224,224), device='CPU:0')
benchmarks = benchmarker.execute()
```

Output will be as following:

```python
{'model': 'resnet50',
 'input_size': '224x224',
 'batch_size': 2,
 'cpu': 'Intel(R) Core(TM) i5-5200U CPU @ 2.20GHz',
 'gpus': '',
 'memory': '15.54GB',
 'os': '#71~20.04.1-Ubuntu SMP Thu Jul 15 17:46:08 UTC 2021',
 'python': '3.9.5',
 'framework': 'TensorFlow 2.5.0',
 'gpu_memory_used': '',
 'benchmark': {'inference_time': 246.02279029786587,
  'throughput_time': 123.01139514893293,
  'std': 6.702327991984426}}
```



## List of Available Models

* InceptionUNet
* SpineNet
* efficientdet-d0 (other possible suffix are d1-d7)
* All models available in Keras applications, [See](https://www.tensorflow.org/api_docs/python/tf/keras/applications#functions_2) list of Keras available models.

## Run with Weights and Biases

If you want to add all the benchmarks results in Weights and Biases (aka wandb), first login to wandb using the terminal

```
wandb login
```

Or you can login in interactive environment when you'll run `benchmarker.execute(wandb=True)`

It will ask you to provide *API key*. Go to the [Authorize](https://wandb.ai/authorize) page to get *API key*. 

Note: You should must have wandb account.

After logging in successfully, execute benchmarks `execute` method with `wandb=True` as following:

```python
# Import the library
import rbm
# Create benchmark instance
benchmarker = rbm.utils.Benchmark(model='ResNet50', batch_size=2, img_size=(224,224), device='CPU:0')
benchmarks = benchmarker.execute(wandb=True)
```

You should see your each benchmarks results in *wandb* project with project name *benchmarks* (by default). You must see 3 charts *std, inference_time* and *throughput_time* and a table with all benchmarks results. Each time you will run benchmarks, your results will append in *benchmarks* project.

![wandb sample image](.wandb_sample.png) 



## Add Custom Model

To calculate the benchmarks for your own custom model, your model should be in following format:

```python
class ModelName:
    def __init__(self, device='cuda:0', **kwargs):
        self.device = device # device argument is necessary if model framework is pytorch, else no need 
        self.__framework__ = 'Framework 2.3.0' # Required: Fullname of framework used for model
        self.__name___ = 'model_name' # Optional: if class name not defining the model name, then optionaly pass the name of model
        self.__type__ = 'cv' # Required: Type of model e.g. cv for computer vision, because data will be generated based on type.
    def __call__(self):
        # This method should build your model
    def preprocess(self, inputs):
        # This is an optional method, if data need to be preprocessed before predictions and not required to add into inference benchmarking.
        return inputs
    def predict(self, inputs):
        # This method should predict the model output
        return predictions
```

There are two ways to execute your custom model.

1. Pass the `ModelName` class instance as following

   ```python
   benchmarker = rbm.utils.Benchmark(model=ModelName, device='CPU:0')
   ```

2. First, add your model under the directory `rbm/models` and assign `@plugins.register` decorator to`ModelName` class which you can import as `from rbm.utils import plugins`. 

   Then pass the model as following:

   ```python
   benchmarker = rbm.utils.Benchmark(model='ModelName', device='CPU:0')
   ```
   
   **Example:** Here is an example of building custom models:
   
   ```python
   # rbm/models/test_model.py
   
   import tensorflow as tf
   from rbm.utils import plugins
   
   @plugins.register
   class TestModel:
       def __init__(self, img_size):
           self.image_size = img_size
           self._model = None
           self.__framework__ = 'Tensorflow ' + tf.__version__
           self.__name__ = 'test-model'
           self.__type__ = 'cv'
       def __call__(self):
           inputs = tf.keras.Input((128, 128, 3))
           outputs = tf.keras.layers.Conv2D(6, (1, 1))(inputs)
           model = tf.keras.Model(inputs, outputs)
           self._model = model
       def predict(self, inputs):
           return self._model.predict(inputs)
   ```
   
   Now we can execute this model benchmarks as:
   
   ```python
   benchmarker = rbm.utils.Benchmark(model='TestModel', batch_size=2, img_size=(224,224), device='CPU:0')
   # batch_size is added for cv type models to generate data for respective batch_size
   benchmarks = benchmarker.execute()
   ```
   
   

