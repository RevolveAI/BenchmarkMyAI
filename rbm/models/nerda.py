

import NERDA as nerda
from NERDA.precooked import EN_ELECTRA_EN, EN_BERT_ML
import nltk
from nltk.tokenize import word_tokenize
import torch
from rbm.utils import plugins

@plugins.register
class NERDA:
    variants = ['EN_ELECTRA_EN', 'EN_BERT_ML']

    def __init__(self, model_name, device='cpu', batch_size=1):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.__name__ = model_name
        self.__framework__ = 'PyTorch ' + torch.__version__
        self.__type__ = 'nlp:ner'
        self._model = None

    def __call__(self, *args, **kwargs):
        model = eval(self.model_name)
        model = model(device=self.device)
        try:
            model.load_network()
        except (AssertionError, RuntimeError):
            model.download_network()
            model.load_network()
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        self._model = model

    def preprocess(self, inputs):
        tokenized = [word_tokenize(_text) for _text in inputs]
        return tokenized

    def predict(self, inputs):
        return self._model.predict(inputs, batch_size=self.batch_size)


