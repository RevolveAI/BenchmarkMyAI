#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 16:47:22 2021

@author: asdkazmi
"""

import tensorflow as tf
import numpy as np
import time
import re
from .. import models
from sys import exc_info
import platform
import cpuinfo
import GPUtil
import psutil
from .wandb import WandB
import os

class Benchmark:
    def __init__(self, model, batch_size, img_size, device='CPU:0'):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.img_size = img_size
    def memoryInfo(self):
        memory_info = tf.config.experimental.get_memory_info(self.device)
        memory_used = dict()
        for _mt in memory_info:
            memory_used.update({_mt+' memory used:': str(round(memory_info[_mt]*1e-6, 2))+'MB'})
            print(_mt+' memory used:', round(memory_info[_mt]*1e-6, 2), 'MB')
        return memory_used
  
    def execute(self, wandb=False):
        with tf.device(self.device):
            if 'CPU' not in self.device:
                print('Before Execution:')
                _ = self.memoryInfo()
            test_batch_images = np.random.uniform(size=(self.batch_size, *self.img_size, 3))
            # if (type(self.model) is str) and ('efficientdet' in self.model):
            #     # from ..models import efficientdet
            #     efficientdet = models.efficientdet
            #     inference = efficientdet.executeInfer(model_name=self.model, batch_size=self.batch_size, image_size=self.img_size)
            #     if not inference[0]:
            #         return inference[1]
            #     inference = inference[1]
            #     inference_time_batch = inference['inference_time_batch']*1000
            #     fps = inference['fps']
            #     std_time = None
            #     framework = inference['framework']
            #     model_name = inference['name']
            # else:
            if type(self.model) is str:
                if re.match('(ResNet|MobileNet|EfficientNet|NASNet){1}.*', self.model):
                    # from ..models import KerasModels
                    KerasModels = models.KerasModels
                    model = KerasModels().load(self.model, self.img_size)
                    assert model[0], model[1]
                    model = model[1]
                    # framework = model.__framework__
                    # model_name = model.__name__
                elif re.match('(efficientdet){1}.*', self.model):
                    model = models.EfficientDet(self.model, self.batch_size)
                    model()
                    test_batch_images = model.preprocess_images(test_batch_images)
                else:
                    try:
                        model = eval(f'models.{self.model}')
                        model = model(img_size=self.img_size)
                        # framework = model.__framework__
                        # model_name = model.__name__
                    except AssertionError:
                        raise AssertionError(exc_info()[1])
                    except AttributeError:
                        raise ValueError("invalid model name")
            else:
                model = self.model
            framework = model.__framework__
            model_name = model.__name__
            for _ in range(10):
                model.predict(test_batch_images)
            time_list = []
            for _ in range(10):
                start_batch = time.perf_counter()
                model.predict(test_batch_images)
                end_batch = time.perf_counter()
                infer = (end_batch - start_batch)*1000
                time_list.append(infer)
            inference_time_batch = sum(time_list) / 10
            std_time = np.std(time_list)
            memory_info = {}
            if 'CPU' not in self.device:
                print('After Execution:')
                memory_info = self.memoryInfo()
                memory_info = {'device': self.device, **memory_info}
            throughput_time = inference_time_batch / self.batch_size
            # print(f'''
            # Inference Time (milliseconds):   {inference_time_batch}
            # Throughput:                     {throughput_time}
            # Standard Deviation of Time:     {std_time}
            # ''')

            def get_size(bytes, suffix="B"):
                factor = 1024
                for unit in ["", "K", "M", "G", "T", "P"]:
                    if bytes < factor:
                        return f"{bytes:.2f}{unit}{suffix}"
                    bytes /= factor
            output = {
                    'model': model_name,
                    'input_size': f'{test_batch_images.shape[1]}x{test_batch_images.shape[2]}',
                    'batch_size': self.batch_size,
                    'cpu': cpuinfo.get_cpu_info()['brand_raw'],
                    'gpus': ' | '.join([gpu.name for gpu in GPUtil.getGPUs()]) if 'CPU' not in self.device else '',
                    'memory': get_size(psutil.virtual_memory().total),
                    'os': platform.version(),
                    'python': platform.python_version(),
                    'framework': framework,
                    'memory_info': memory_info,
                    'benchmark': {
                        'inference_time': inference_time_batch,
                        'throughput_time': throughput_time,
                        'std': std_time
                        }
                    }
            if wandb:
                run_name = f'{model_name} {self.img_size[0]}x{self.img_size[1]} {self.batch_size}'
                wandb_instance = WandB(project_name='benchmarks', run_name=run_name)
                wandb_instance.init()
                wandb_instance.plot_and_table(benchmarks=output)
                wandb_instance.close()
            return output