import re
from pathlib import Path
from typing import Dict, Type, Union

import torch
from transformers.tokenization_utils import (PreTrainedTokenizer,
                                             PreTrainedTokenizerFast)

from dataset import BaseDataset
from model import CorrectionModel
from pretrained import PRETRAINED_MODELS
from utils import get_last_pretrained_weight_path, load_params


class Predictor:
    def __init__(self,
                 model_name: str,
                 models_root: Path = Path("models"),
                 dataset_class: Type[BaseDataset] = BaseDataset,
                 model_weights: str = None,
                 ) -> None:

        model_dir = models_root / model_name
        self.params = load_params(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not model_weights:
            self.weights = get_last_pretrained_weight_path(model_dir)
        else:
            self.weights = model_dir / 'weights' / model_weights

        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.dataset_class = dataset_class

    def load_model(self) -> CorrectionModel:
        model = CorrectionModel(self.params['pretrained_model'],
                                self.params['targets'],
                                self.params['freeze_pretrained'],
                                self.params['lstm_dim'])

        model.to(self.device)
        model.load(self.weights, map_location=self.device)
        model.eval()
        return model

    def load_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        name = self.params['pretrained_model']
        tokenizer = PRETRAINED_MODELS[name][1].from_pretrained(name)
        return tokenizer

    def __call__(self, text: str, decode_map: Dict[int, str] = {0: '', 1: ',', 2: '.'}) -> str:
        words_original_case = text.split()
        tokens = text.split()
        result = ""

        token_style = PRETRAINED_MODELS[self.params['pretrained_model']][3]
        seq_len = self.params['sequence_length']
        decode_idx = 0

        data = self.dataset_class.parse_tokens(tokens,
                                               self.tokenizer,
                                               seq_len,
                                               token_style,
                                               debug=False)

        data = torch.tensor(data)

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
        # result = re.sub(r'(\w)[\.|,]*$', r'\g<1>.', result)
        return result


if __name__ == '__main__':
    inputs = [
        'отфильтруй по убыванию средней стоимости сгруппируй по возрасту',
        'сгруппируй по коду отсортируй по убыванию средней выручки',
        'сгруппируй по коду отсортируй по убыванию средней выручки и покажи только те которые начинаются на единицу',
        'покажи сотрудников которые закрыли больше десяти задач',
        'чтобы изучить SQL необходима практика',
        'поэтому даже несмотря на то что SQL старый язык похоже его ждет еще очень долгая жизнь и блестящее будущее'
    ]

    punct_restorer = Predictor('repunct-model^4')

    for input in inputs:
        output = punct_restorer(input)
        print(f'Original text:\n{input}')
        print(f'Punctuated text:\n{output}\n')
