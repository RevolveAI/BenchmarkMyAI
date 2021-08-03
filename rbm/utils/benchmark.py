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
    def __init__(self, model, batch_size, img_size, device='CPU:0', **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.img_size = img_size
        self.kwargs = kwargs
    def memoryInfo(self):
        memory_info = tf.config.experimental.get_memory_info(self.device)
        memory_used = dict()
        for _mt in memory_info:
            memory_used.update({_mt: str(round(memory_info[_mt]*1e-6, 2))+'MB'})
            print(_mt+' memory used:', round(memory_info[_mt]*1e-6, 2), 'MB')
        return memory_used

    def load_model(self):
        if type(self.model) is str:
            if re.match('(efficientdet){1}.*', self.model):
                model = models.EfficientDet(self.model, self.batch_size)
                model()
            else:
                try:
                    model = eval(f'models.{self.model}')
                    model = model(img_size=self.img_size, **self.kwargs)
                    model()
                except AttributeError:
                    try:
                        model = models.KerasModels.load(self.model, self.img_size, **self.kwargs)
                    except AttributeError:
                        raise ValueError("invalid model name")
        else:
            model = self.model
        return model

    def _calculate_benchmarks(self, model, inputs):
        with tf.device(self.device):
            for _ in range(5):
                model.predict(inputs)
            time_list = []
            for _ in range(10):
                start_batch = time.perf_counter()
                model.predict(inputs)
                end_batch = time.perf_counter()
                infer = (end_batch - start_batch)*1000
                time_list.append(infer)
            inference_time_batch = sum(time_list) / 10
            std_time = np.std(time_list)
        throughput_time = inference_time_batch / self.batch_size
        return {
            'inference_time': inference_time_batch,
            'throughput_time': throughput_time,
            'std': std_time
        }

    def _wandb(self, project_name, model_name, output):
        run_name = f"{model_name} {output['input_size']} {self.batch_size}"
        wandb_instance = WandB(project_name=project_name, run_name=run_name)
        wandb_instance.init()
        wandb_instance.plot_and_table(benchmarks=output)
        wandb_instance.close()

    def execute(self, wandb=False, project_name='benchmarks'):
        if 'CPU' not in self.device:
            print('Before Execution:')
            _ = self.memoryInfo()
        test_batch_images = np.random.uniform(size=(self.batch_size, *self.img_size, 3))
        model = self.load_model()
        try:
            test_batch_images = model.preprocess_images(test_batch_images)
        except:
            pass
        framework = model.__framework__
        model_name = model.__name__
        benchmarks = self._calculate_benchmarks(model=model, inputs=test_batch_images)
        gpu_memory_used = ''
        if 'CPU' not in self.device:
            print('After Execution:')
            memory_info = self.memoryInfo()
            gpu_memory_used = memory_info['peak']
            # memory_info = {'device': self.device, **memory_info}

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
                'gpu_memory_used': gpu_memory_used,
                'benchmark': benchmarks
                }
        if wandb:
            self._wandb(project_name=project_name, model_name=model_name, output=output)
        return output