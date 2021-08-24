

from transformers import AutoModelForSequenceClassification
from rbm.utils import plugins
from rbm.models.nlp.hugging_face import HuggingFace
from rbm.backends.nlp import TextClassification


@plugins.register
class HuggingFaceTC(HuggingFace, TextClassification):
    variants = ['distilbert-base-uncased-finetuned-sst-2-english', 'textattack/bert-base-uncased-imdb']

    def __init__(self, model_name, batch_size=1, **kwargs):
        HuggingFace.__init__(self, model_name=model_name, **kwargs)
        TextClassification.__init__(self, batch_size=batch_size)

    def __call__(self):
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)

    def _postprocess(self, outputs):
        return outputs[0].softmax(1)





