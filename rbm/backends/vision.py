

import numpy as np
from rbm.utils.info import get_models_info


class ImageProcessing:

    def __init__(self, batch_size, img_size, *args, **kwargs):
        self.batch_size = batch_size
        self.img_size = img_size
        get_models_info(self)

    def generate_data(self):
        data = np.random.uniform(size=(self.batch_size, *self.img_size, 3))
        return data

    def data_shape(self, data):
        shape = {'batch_size': self.batch_size,
                 'input_size': f'{data.shape[1]}x{data.shape[2]}'}
        return shape


