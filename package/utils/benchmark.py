#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 16:47:22 2021

@author: asdkazmi
"""

import tensorflow as tf
import numpy as np
import time

class benchmark:
    def __init__(self, model, batch_size, img_size, gpu_device=None):
        self.model = model
        self.batch_size = batch_size
        self.device = gpu_device
        self.img_size = img_size
        
    def memoryInfo(self):
        memory_info = tf.config.experimental.get_memory_info(self.device)
        memory_used = dict()
        for _mt in memory_info:
          memory_used.update({_mt+' memory used:': str(round(memory_info[_mt]*(1e-6), 2))+'MB'})
          print(_mt+' memory used:', round(memory_info[_mt]*(1e-6), 2), 'MB')
        return memory_used
  
    def execute(self):
        test_batch_images = np.random.uniform(size=(self.batch_size, *self.img_size, 3))
        if self.device is not None:    
            print('Befor Execution:')
            _ = self.memoryInfo()
        else:
            print('Testing on CPU')
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
        memory_info = {}
        if self.device is not None:
            print('After Execution:')
            memory_info = self.memoryInfo()
            memory_info = {'device': self.device, **memory_info}
        inference_time_batch = sum(time_list) / 10
        std_time = np.std(time_list)
        throughput_time = inference_time_batch / self.batch_size
        print(f'''
        Inference Time (miliseconds):   {round(inference_time_batch, 2)}
        Throughput:                     {round(throughput_time, 2)}
        Standard Deviation of Time:     {round(std_time, 2)}
        ''')
        return {'inference_time': inference_time_batch, 
                'throughput_time': throughput_time, 
                'std': std_time,
                **memory_info
                }
    