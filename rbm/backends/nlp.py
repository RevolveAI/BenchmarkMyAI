

from rbm.utils.info import get_models_info


class NLP:

    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size
        get_models_info(self)

    def data_shape(self, data):
        shape = {'batch_size': self.batch_size}
        return shape


class QuestionAnswering(NLP):

    def __init__(self, batch_size, *args, **kwargs):
        super().__init__(batch_size=batch_size)

    def generate_data(self):
        data = {
            'question': ["What was President Donald Trump's prediction?"] * self.batch_size,
            'context': ["The US has passed the peak on new coronavirus cases, \
President Donald Trump said and predicted that some states would reopen this month.\
The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any \
country in the world."] * self.batch_size
        }
        return data


class NER(NLP):

    def __init__(self, batch_size, *args, **kwargs):
        super().__init__(batch_size=batch_size)

    def generate_data(self):
        data = ['Old MacDonald had a farm'] * self.batch_size
        return data


class TextClassification(NLP):

    def __init__(self, batch_size, *args, **kwargs):
        super().__init__(batch_size=batch_size)

    def generate_data(self):
        data = [['It seems to me that I can make everything impossible to possible'] * self.batch_size]
        return data



