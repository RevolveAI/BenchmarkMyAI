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

    def __init__(self, model, device=None, **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs


    @staticmethod
    def list_models():
        return models.models_names()

    def load_model(self):
        if type(self.model) is str:
            exited_models = models.models_names()
            if self.model in exited_models:
                model = models.load(self.model, device=self.device, **self.kwargs)
                model()
            else:
                raise ValueError('invalid model name')
        else:
            model = self.model(**self.kwargs)
            model()
        return model

    def _calculate_benchmarks(self, model, inputs):
        with model.device_placement():
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
        run_name = f"{model_name} {output.get('input_size', '')} {output.get('batch_size', '')}"
        wandb_instance = WandB(project_name=project_name, run_name=run_name)
        wandb_instance.init()
        wandb_instance.plot_and_table(benchmarks=output)
        wandb_instance.close()

    def execute(self, wandb=False, project_name='benchmarks'):
        model = self.load_model()
        model_type = model.__type__
        framework = model.__framework__
        model_name = model.__name__
        data = model.generate_data()
        if hasattr(model, 'preprocess'):
            data = model.preprocess(data)
        benchmarks = self._calculate_benchmarks(model=model, inputs=data)
        gpu_memory_used = model.memory_info()

        def get_size(bytes, suffix="B"):
            factor = 1024
            for unit in ["", "K", "M", "G", "T", "P"]:
                if bytes < factor:
                    return f"{bytes:.2f}{unit}{suffix}"
                bytes /= factor
        output = {
                'model': model_name,
                'type': model_type,
                **model.data_shape(data),
                'cpu': cpuinfo.get_cpu_info()['brand_raw'],
                'gpus': ' | '.join([gpu.name for gpu in GPUtil.getGPUs()]) if gpu_memory_used != '' else '',
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


