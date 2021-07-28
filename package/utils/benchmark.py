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
  
    def execute(self):
        with tf.device(self.device):
            if 'CPU' not in self.device:
                print('Before Execution:')
                _ = self.memoryInfo()
            if (type(self.model) is str) and ('efficientdet' in self.model):
                # from ..models import efficientdet
                efficientdet = models.efficientdet
                inference = efficientdet.executeInfer(model_name=self.model, batch_size=self.batch_size, image_size=self.img_size)
                if not inference[0]:
                    return inference[1]
                inference = inference[1]
                inference_time_batch = inference['inference_time_batch']*1000
                fps = inference['fps']
                std_time = None
            else:
                if type(self.model) is str:
                    if re.match('(ResNet|MobileNet|EfficientNet|NASNet){1}.*', self.model):
                        # from ..models import KerasModels
                        KerasModels = models.KerasModels
                        model = KerasModels().load(self.model, self.img_size)
                        assert model[0], model[1]
                        model = model[1]
                    else:
                        try:
                            model = eval(f'models.{self.model}')
                            model = model(img_size=self.img_size)
                        except AssertionError:
                            raise AssertionError(exc_info()[1])
                        except:
                            raise ValueError("invalid model name")
                else:
                    model = self.model
                test_batch_images = np.random.uniform(size=(self.batch_size, *self.img_size, 3))
                for _ in range(2):
                    model.predict(test_batch_images)
                time_list = []
                for _ in range(10):
                    start_batch = time.perf_counter()
                    model.predict(test_batch_images)
                    # print('completed execution')
                    end_batch = time.perf_counter()
                    infer = (end_batch - start_batch)*1000
                    # print('Inference Time of One Iteration:', infer)
                    time_list.append(infer)
                inference_time_batch = sum(time_list) / 10
                std_time = np.std(time_list)
            memory_info = {}
            if 'CPU' not in self.device:
                print('After Execution:')
                memory_info = self.memoryInfo()
                memory_info = {'device': self.device, **memory_info}
            throughput_time = inference_time_batch / self.batch_size
            print(f'''
            Inference Time (milliseconds):   {inference_time_batch}
            Throughput:                     {throughput_time}
            Standard Deviation of Time:     {std_time}
            ''')
            return {'inference_time': inference_time_batch, 
                    'throughput_time': throughput_time, 
                    'std': std_time,
                    # 'inference_time_iters': time_list,
                    **memory_info
                    }
    