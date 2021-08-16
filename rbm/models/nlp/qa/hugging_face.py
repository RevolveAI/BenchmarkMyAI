

from transformers import AutoModelForQuestionAnswering
from rbm.utils import plugins
from rbm.models.nlp.hugging_face import HuggingFace
from rbm.backends.nlp import QuestionAnswering



@plugins.register
class HuggingFaceQA(HuggingFace, QuestionAnswering):
    variants = ['PremalMatalia/albert-base-best-squad2', 'elgeish/cs224n-squad2.0-albert-base-v2',
                'madlag/albert-base-v2-squad', 'twmkn9/albert-base-v2-squad2',
                'twmkn9/distilbert-base-uncased-squad2', 'distilbert-base-uncased-distilled-squad']
    
    def __init__(self, model_name, batch_size=1, **kwargs):
        HuggingFace.__init__(self, model_name=model_name, **kwargs)
        QuestionAnswering.__init__(self, batch_size=batch_size)

    def __call__(self):
        self._model = AutoModelForQuestionAnswering.from_pretrained(self.model_name).to(self.device)





