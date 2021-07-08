from typing import Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
from torch.tensor import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from neuro_comma.augmentation import AUGMENTATIONS
from neuro_comma.pretrained import TOKEN_IDX


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 targets: Dict[str, int],
                 sequence_len: int,
                 token_style: str,
                 *args,
                 **kwargs) -> None:

        self.tokenizer = tokenizer
        self.targets = targets
        self.seq_len = sequence_len
        self.token_style = token_style

        if isinstance(files, list):
            self.data = []
            for file in files:
                self.data += self._parse_data(file, *args, **kwargs)
        else:
            self.data = self._parse_data(files, *args, **kwargs)

    def _parse_data(self, file_path: str, *args, **kwargs) -> List[List[List[int]]]:
        """Parse file to train data

        Args:
            file_path (`str`): text file path that contains tokens and punctuations separated by tab in lines
        Returns:
            list[Batch]: each having sequence_len punctuation_mask is used to ignore special indices like padding and intermediate sub-word token during evaluation
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            x, y = [], []
            for i, line in enumerate(file):
                if (line := line.strip()):
                    token = line.rsplit('\t', 1)
                    if len(token) == 2:
                        x.append(token[0])
                        target = self.targets[token[1]]
                        y.append(target)
                    else:
                        continue

        data = self.parse_tokens(x, self.tokenizer, self.seq_len, self.token_style, y, *args, **kwargs)
        return data

    @classmethod
    def parse_tokens(cls,
                     tokens: Union[List[str], Tuple[str]],
                     tokenizer: PreTrainedTokenizer,
                     seq_len: int,
                     token_style: str,
                     targets: Optional[List[int]] = None,
                     *args,
                     **kwargs) -> List[List[List[int]]]:
        """
        Convert tokenized data for model prediction

        Args:
            tokens (`Union[list[str], tuple[str]]`): splited tokens
            tokenizer (`PreTrainedTokenizer`): tokenizer which split tokens to subtokens
            seq_len (`int`): sequence length
            token_style (`str`): token_style from pretrained.TOKEN_IDX

        Returns:
            (`list[BatchWithoutTarget]`): list of bathces

        ```txt
        tokens    : [token  token  ##token  PAD ]
             x    : [321    1233   23121    101 ]
             y    : [tar    0      tar      0   ]
        y_mask    : [1      0      1        0   ]
        attn_mask : [1      1      1        0   ]
        ```

        """
        data_items = []
        # loop until end of the entire text
        idx = 0

        debug = kwargs.get('debug')
        if debug:
            pbar = tqdm(total=len(tokens))

        while idx < len(tokens):
            x = [TOKEN_IDX[token_style]['START_SEQ']]
            w_id = [-1]    # word indexes
            y = [0]
            y_mask = [1] if targets else [0]

            # loop until we have required sequence length
            # -1 because we will have a special end of sequence token at the end
            while len(x) < seq_len - 1 and idx < len(tokens):
                word_pieces = tokenizer.tokenize(tokens[idx])

                # if taking these tokens exceeds sequence length we finish
                # current sequence with padding
                # then start next sequence from this token
                if len(word_pieces) + len(x) >= seq_len:
                    break
                for i in range(len(word_pieces) - 1):
                    x.append(tokenizer.convert_tokens_to_ids(word_pieces[i]))
                    w_id.append(idx)
                    y.append(0)
                    y_mask.append(0)
                if len(word_pieces) > 0:
                    x.append(tokenizer.convert_tokens_to_ids(word_pieces[-1]))
                else:
                    x.append(TOKEN_IDX[token_style]['UNK'])

                w_id.append(idx)

                if targets:
                    y.append(targets[idx])
                else:
                    y.append(0)

                y_mask.append(1)

                idx += 1
                if debug:
                    pbar.update(1)

            x.append(TOKEN_IDX[token_style]['END_SEQ'])
            w_id.append(-1)
            y.append(0)
            if targets:
                y_mask.append(1)
            else:
                y_mask.append(0)

            # Fill with pad tokens
            if len(x) < seq_len:
                x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(seq_len - len(x))]
                w_id = w_id + [-100 for _ in range(seq_len - len(w_id))]
                y = y + [0 for _ in range(seq_len - len(y))]
                y_mask = y_mask + [0 for _ in range(seq_len - len(y_mask))]

            attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]

            data_items.append([x, w_id, attn_mask, y, y_mask])

        if debug:
            pbar.close()

        return data_items

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.data[index][0]
        attn_mask = self.data[index][2]
        y = self.data[index][3]
        y_mask = self.data[index][4]

        x = torch.tensor(x)                     # type: ignore
        attn_mask = torch.tensor(attn_mask)     # type: ignore
        y = torch.tensor(y)                     # type: ignore
        y_mask = torch.tensor(y_mask)           # type: ignore

        return x, y, attn_mask, y_mask          # type: ignore


class RepunctDataset(BaseDataset):
    def __init__(self,
                 files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 targets: Dict[str, int],
                 sequence_len: int,
                 token_style: str,
                 is_train=False,
                 augment_rate=0.,
                 augment_type='substitute',
                 *args,
                 **kwargs) -> None:
        """Preprocess data for restore punctuation

        Args:
            files (`Union[str, list[str]]`): single file or list of text files containing tokens and punctuations separated by tab in lines
            tokenizer (`PreTrainedTokenizer`): tokenizer that will be used to further tokenize word for BERT like models
            targets (`dict[str, int]`): dict with targets
            sequence_len (`int`): length of each sequence
            token_style (`str`): For getting index of special tokens in pretrained.TOKEN_IDX
            is_train (`bool, optional`): if false do not apply augmentation. Defaults to False.
            augment_rate (`float, optional`): percent of data which should be augmented. Defaults to 0.0.
            augment_type (`str, optional`): augmentation type. Defaults to 'substitute'.
        """
        super().__init__(files, tokenizer, targets, sequence_len, token_style, *args, **kwargs)

        self.is_train = is_train
        self.augment_type = augment_type
        self.augment_rate = augment_rate

    def _augment(self, x, y, y_mask):
        x_aug = []
        y_aug = []
        y_mask_aug = []
        for i in range(len(x)):
            r = np.random.rand()
            if r < self.augment_rate:
                AUGMENTATIONS[self.augment_type](x, y, y_mask, x_aug, y_aug, y_mask_aug, i, self.token_style)
            else:
                x_aug.append(x[i])
                y_aug.append(y[i])
                y_mask_aug.append(y_mask[i])

        if len(x_aug) > self.seq_len:
            # len increased due to insert
            x_aug = x_aug[:self.seq_len]
            y_aug = y_aug[:self.seq_len]
            y_mask_aug = y_mask_aug[:self.seq_len]
        elif len(x_aug) < self.seq_len:
            # len decreased due to delete
            x_aug = x_aug + [TOKEN_IDX[self.token_style]['PAD'] for _ in range(self.seq_len - len(x_aug))]
            y_aug = y_aug + [0 for _ in range(self.seq_len - len(y_aug))]
            y_mask_aug = y_mask_aug + [0 for _ in range(self.seq_len - len(y_mask_aug))]

        attn_mask = [1 if token != TOKEN_IDX[self.token_style]['PAD'] else 0 for token in x]
        return x_aug, y_aug, attn_mask, y_mask_aug

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.data[index][0]
        attn_mask = self.data[index][2]
        y = self.data[index][3]
        y_mask = self.data[index][4]

        if self.is_train and self.augment_rate > 0:
            x, y, attn_mask, y_mask = self._augment(x, y, y_mask)

        x = torch.tensor(x)                     # type: ignore
        attn_mask = torch.tensor(attn_mask)     # type: ignore
        y = torch.tensor(y)                     # type: ignore
        y_mask = torch.tensor(y_mask)           # type: ignore

        return x, y, attn_mask, y_mask          # type: ignore
