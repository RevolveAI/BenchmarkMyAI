#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 16:47:22 2021

@author: asdkazmi
"""

import tensorflow as tf
import numpy as np
import time


class Benchmark:
    def __init__(self, model, batch_size, img_size, gpu_device=None):
        self.model = model
        self.batch_size = batch_size
        self.device = gpu_device
        self.img_size = img_size
    def memory_info(self):
        memory_info = tf.config.experimental.get_memory_info(self.device)
        memory_used = dict()
        for _mt in memory_info:
            memory_used.update({_mt+' memory used:': str(round(memory_info[_mt]*1e-6, 2))+'MB'})
            print(_mt+' memory used:', round(memory_info[_mt]*1e-6, 2), 'MB')
        return memory_used
  
    def execute(self):
        if self.device is not None:    
            print('Before Execution:')
            _ = self.memoryInfo()
        else:
            print('Testing on CPU')
        if (type(self.model) is str) and ('efficientdet' in self.model):
            from ..models import efficientdetD0
            inference = efficientdetD0.executeInfer(model_name=self.model, batch_size=self.batch_size)
            inference_time_batch = inference['inference_time_batch']*1000
            fps = inference['fps']
            std_time = None
        else:
            test_batch_images = np.random.uniform(size=(self.batch_size, *self.img_size, 3))
            for _ in range(2):
                self.model.predict(test_batch_images[:1])
            time_list = []
            for _ in range(10):
                start_batch = time.perf_counter()
                self.model.predict(test_batch_images)
                # print('completed execution')
                end_batch = time.perf_counter()
                infer = (end_batch - start_batch)*1000
                # print('Inference Time of One Iteration:', infer)
                time_list.append(infer)
            inference_time_batch = sum(time_list) / 10
            std_time = np.std(time_list)
        memory_info = {}
        if self.device is not None:
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
    