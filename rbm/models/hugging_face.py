

import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
import torch
from rbm.utils import plugins


class HuggingFace:

    def __init__(self, model_name, model_type, device='cpu:0', batch_size=None, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, return_token_type_ids=True)
        self._model = None
        self._inputs = None
        self.__framework__ = 'Pytorch ' + torch.__version__ + ' | HuggingFace Transformers ' + transformers.__version__
        self.__name__ = model_name
        self.__type__ = model_type

    def _preprocess(self, inputs):
        if type(inputs) is dict:
            inputs = list(inputs.values())
        inputs = self._tokenizer(*inputs, return_tensors='pt', padding=True).to(self.device)
        self._inputs = inputs
        return inputs

    def predict(self, inputs):
        inputs = inputs.copy()
        inputs = self._preprocess(inputs=inputs)
        preds = self._model(**inputs, **self.kwargs)  # , start_positions=start_positions, end_positions=end_positions
        answer = self._postprocess(preds)
        return answer

    def _postprocess(self, outputs):
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        answer_tokens = list()
        answers = list()
        for i in range(len(start_scores)):
            ids_tokens = self._tokenizer.convert_ids_to_tokens(self._inputs['input_ids'].cpu().numpy()[i])
            answer_tokens.append(ids_tokens[torch.argmax(start_scores[i]): torch.argmax(end_scores[i]) + 1])
            answers.append(self._tokenizer.convert_tokens_to_string(answer_tokens[i]))
        return answers


@plugins.register
class HuggingFaceQA(HuggingFace):
    variants = ['PremalMatalia/albert-base-best-squad2', 'elgeish/cs224n-squad2.0-albert-base-v2',
                'madlag/albert-base-v2-squad', 'twmkn9/albert-base-v2-squad2',
                'twmkn9/distilbert-base-uncased-squad2', 'distilbert-base-uncased-distilled-squad']
    
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name=model_name, model_type='nlp:qa', **kwargs)

    def __call__(self):
        self._model = AutoModelForQuestionAnswering.from_pretrained(self.model_name).to(self.device)


@plugins.register
class HuggingFaceTC(HuggingFace):
    variants = ['distilbert-base-uncased-finetuned-sst-2-english', 'textattack/bert-base-uncased-imdb']

    def __init__(self, model_name, **kwargs):
        super().__init__(model_name=model_name, model_type='nlp:tc', **kwargs)

    def __call__(self):
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)

    def _postprocess(self, outputs):
        return outputs[0].softmax(1)





