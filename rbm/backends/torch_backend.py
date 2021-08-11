

import torch
from contextlib import contextmanager


class TorchBackend:

    def __init__(self, device=None, *args, **kwargs):
        if device is None:
            self.get_preferred_device()
        else:
            self.device = torch.device(device.lower().replace('gpu', 'cuda'))
        self.__framework__ = 'PyTorch ' + torch.__version__

    def get_preferred_device(self):
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{torch.cuda.current_device()}')
        else:
            self.device = torch.device('cpu')

    def memory_info(self):
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(self.device)
            memory_used = str(round(memory_used * 1e-6, 2)) + 'MB'
        else:
            memory_used = ''
        return memory_used

    @contextmanager
    def device_placement(self):
        yield



