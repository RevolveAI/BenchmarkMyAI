

from rbm.backends import TorchBackend
import transformers
from transformers import AutoTokenizer
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class HuggingFace(TorchBackend):

    def __init__(self, model_name, device, **kwargs):
        TorchBackend.__init__(self, device=device)
        self.model_name = model_name
        self.kwargs = kwargs
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, return_token_type_ids=True)
        self._model = None
        self._inputs = None
        self.__framework__ = self.__framework__ + ' | HuggingFace Transformers ' + transformers.__version__
        self.__name__ = model_name

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



