from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from neuro_comma.dataset import BaseDataset
from neuro_comma.model import CorrectionModel
from neuro_comma.pretrained import PRETRAINED_MODELS
from neuro_comma.utils import get_last_pretrained_weight_path, load_params


class BasePredictor:
    def __init__(self,
                 model_name: str,
                 models_root: Path = Path("models"),
                 dataset_class: Type[BaseDataset] = BaseDataset,
                 model_weights: Optional[str] = None,
                 quantization: Optional[bool] = False,
                 *args,
                 **kwargs,
                 ) -> None:

        model_dir = models_root / model_name
        self.params = load_params(model_dir)
        self.device = torch.device('cuda' if (not quantization) and torch.cuda.is_available() else 'cpu')

        if not model_weights:
            self.weights = get_last_pretrained_weight_path(model_dir)
        else:
            self.weights = model_dir / 'weights' / model_weights

        self.model = self.load_model(quantization=quantization)
        self.tokenizer = self.load_tokenizer()
        self.dataset_class = dataset_class

    def load_model(self, quantization: Optional[bool] = False) -> CorrectionModel:
        model = CorrectionModel(self.params['pretrained_model'],
                                self.params['targets'],
                                self.params['freeze_pretrained'],
                                self.params['lstm_dim'])

        if quantization:
            model = model.quantize()

        model.to(self.device)
        model.load(self.weights, map_location=self.device)
        model.eval()
        return model

    def load_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        name = self.params['pretrained_model']
        tokenizer = PRETRAINED_MODELS[name][1].from_pretrained(name)
        return tokenizer


class RepunctPredictor(BasePredictor):
    def __call__(self, text: str, decode_map: Dict[int, str] = {0: '', 1: ',', 2: '.'}) -> str:
        words_original_case = text.split()
        tokens = text.split()
        result = ""

        token_style = PRETRAINED_MODELS[self.params['pretrained_model']][3]
        seq_len = self.params['sequence_length']
        decode_idx = 0

        data = torch.tensor(self.dataset_class.parse_tokens(tokens,
                                                            self.tokenizer,
                                                            seq_len,
                                                            token_style))

        x_indecies = torch.tensor([0])
        x = torch.index_select(data, 1, x_indecies).reshape(2, -1).to(self.device)

        attn_mask_indecies = torch.tensor([2])
        attn_mask = torch.index_select(data, 1, attn_mask_indecies).reshape(2, -1).to(self.device)

        y_indecies = torch.tensor([4])
        y_mask = torch.index_select(data, 1, y_indecies).view(-1)

        with torch.no_grad():
            y_predict = self.model(x, attn_mask)

        y_predict = y_predict.view(-1, y_predict.shape[2])
        y_predict = torch.argmax(y_predict, dim=1).view(-1)

        for i in range(y_mask.shape[0]):
            if y_mask[i] == 1:
                result += words_original_case[decode_idx]
                result += decode_map[y_predict[i].item()]
                result += ' '
                decode_idx += 1

        result = result.strip()
        return result
