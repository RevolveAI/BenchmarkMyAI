#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 16:47:22 2021

@author: asdkazmi
"""

import tensorflow as tf
import numpy as np
import time
from .. import models
import platform
import cpuinfo
import GPUtil
import psutil
from .wandb import WandB
import torch
from contextlib import contextmanager


class Benchmark:

    def __init__(self, model, device='CPU:0', **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs


    def memory_info(self):
        if 'cuda' in self.device:
            memory_used = torch.cuda.memory_allocated(self.device)
        elif 'GPU' in self.device:
            memory_used = tf.config.experimental.get_memory_info(self.device)['peak']
        else:
            memory_used = ''
        if memory_used != '':
            memory_used = str(round(memory_used * 1e-6, 2)) + 'MB'
        return memory_used

    @staticmethod
    def list_models():
        return models.models_names()

    def load_model(self):
        if type(self.model) is str:
            exited_models = models.models_names()
            if self.model in exited_models:
                model = models.models(self.model, **self.kwargs)
                if 'torch' in model.__framework__:
                    self.device = self.device.lower().replace('gpu', 'cuda')
                    model.device = torch.device(self.device)
                model()
            else:
                raise ValueError('invalid model name')
        else:
            model = self.model(**self.kwargs)
            model()
        return model

    def generate_data(self, model_type, **kwargs):
        if model_type == 'cv':
            data = np.random.uniform(size=(self.kwargs.get('batch_size', 1), *kwargs.get('img_size', (224, 224)), 3))
        elif model_type == 'nlp:qa':
            data = {
                'context': ["The US has passed the peak on new coronavirus cases, \
President Donald Trump said and predicted that some states would reopen this month.\
The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any \
country in the world."]*self.kwargs.get('batch_size', 1),
                'question': ["What was President Donald Trump's prediction?"]*self.kwargs.get('batch_size', 1)
            }
        else:
            data = None
        return data

    def _data_shape(self, data, model_type):
        shape = {'batch_size': self.kwargs.get('batch_size', 1)}
        if model_type == 'cv':
            shape = shape.update({'input_size': f'{data.shape[1]}x{data.shape[2]}'})
        return shape

    @contextmanager
    def device_placement(self, framework):
        if 'tensorflow' in framework.lower():
            with tf.device(self.device):
                yield
        else:
            yield

    def _calculate_benchmarks(self, model, inputs):
        with self.device_placement(model.__framework__):
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
        throughput_time = inference_time_batch / self.kwargs.get('batch_size', 1)
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
        _ = self.memory_info()
        model = self.load_model()
        model_type = model.__type__
        framework = model.__framework__
        model_name = model.__name__
        dg_kwargs = {}
        if hasattr(model, 'img_size'):
            dg_kwargs = {'img_size': model.img_size}
        data = self.generate_data(model_type=model_type, **dg_kwargs)
        if hasattr(model, 'preprocess'):
            data = model.preprocess(data)
        benchmarks = self._calculate_benchmarks(model=model, inputs=data)
        gpu_memory_used = self.memory_info()

        def get_size(bytes, suffix="B"):
            factor = 1024
            for unit in ["", "K", "M", "G", "T", "P"]:
                if bytes < factor:
                    return f"{bytes:.2f}{unit}{suffix}"
                bytes /= factor
        output = {
                'model': model_name,
                **self._data_shape(data=data, model_type=model_type),
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


