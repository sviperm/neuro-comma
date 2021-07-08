from contextlib import ExitStack
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
from labml import experiment, tracker
from numpy import ndarray
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuro_comma import augmentation
from neuro_comma.argparser import parse_train_arguments
from neuro_comma.dataset import RepunctDataset
from neuro_comma.logger import (log_args, log_target_test_metrics,
                                log_test_metrics, log_text, log_train_epoch,
                                log_val_epoch)
from neuro_comma.model import CorrectionModel
from neuro_comma.pretrained import PRETRAINED_MODELS
from neuro_comma.utils import (export_params, get_last_epoch_params,
                               get_last_pretrained_weight_path,
                               get_model_save_path, load_params, save_weights)

# https://github.com/pytorch/pytorch/issues/11201
torch.multiprocessing.set_sharing_strategy('file_system')

args = parse_train_arguments()

# for reproducibility
if args.seed:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

models_root = Path(args.save_dir)
model_save_path = get_model_save_path(models_root, args)


# MODEL
print('Loading model...')
TARGETS = args.targets
DEVICE = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
MODEL_SAVE_NAME = model_save_path.stem

if not (args.resume or args.fine_tune):
    MODEL = CorrectionModel(args.pretrained_model,
                            targets=TARGETS,
                            freeze_pretrained=args.freeze_pretrained,
                            lstm_dim=args.lstm_dim)
else:
    orig_model_dir = models_root / args.model_name
    orig_params = load_params(orig_model_dir)

    MODEL = CorrectionModel(pretrained_model=orig_params['pretrained_model'],
                            targets=orig_params['targets'],
                            freeze_pretrained=args.freeze_pretrained,
                            lstm_dim=orig_params['lstm_dim'])

    pretrained_weights = get_last_pretrained_weight_path(orig_model_dir)
    MODEL.load(pretrained_weights)

    if args.fine_tune and (len(orig_params['targets']) != len(TARGETS)):
        MODEL.modify_last_linear(in_features=MODEL.hidden_size * 2,
                                 out_features=len(TARGETS))

MODEL.to(DEVICE)

WEIGHTS = torch.FloatTensor(args.weights).to(DEVICE) if args.weights else None
CRITERION = nn.CrossEntropyLoss(weight=WEIGHTS)
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=args.lr, weight_decay=args.decay)
print('Model was loaded.')

# TOKENIZER
print('Loading tokenizer...')
tokenizer = PRETRAINED_MODELS[args.pretrained_model][1].from_pretrained(args.pretrained_model)
token_style = PRETRAINED_MODELS[args.pretrained_model][3]
SEQ_LEN = args.sequence_length
print('Tokenizer was loaded.')

# CONFIG AUGMENTATION
AUG_RATE = args.augment_rate
AUG_TYPE = args.augment_type
augmentation.tokenizer = tokenizer
augmentation.sub_style = args.sub_style
augmentation.alpha_sub = args.alpha_sub
augmentation.alpha_del = args.alpha_del


# DATASETS
data_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': 1
}

print('Loading train data...')
train_dataset = RepunctDataset(args.train_data, tokenizer=tokenizer, targets=TARGETS,
                               sequence_len=SEQ_LEN, token_style=token_style,
                               is_train=True, augment_rate=AUG_RATE,
                               augment_type=AUG_TYPE, debug=True)
train_loader = DataLoader(train_dataset, **data_loader_params)

print('Loading validation data...')
val_dataset = RepunctDataset(args.val_data, tokenizer=tokenizer, targets=TARGETS,
                             sequence_len=SEQ_LEN, token_style=token_style,
                             is_train=True, debug=True)
val_loader = DataLoader(val_dataset, **data_loader_params)

if args.test_data:
    print('Loading test data...')
    if args.test_data == args.val_data:
        test_dataset = val_dataset
        test_loader = val_loader
    else:
        test_dataset = RepunctDataset(args.test_data, tokenizer=tokenizer, targets=TARGETS,
                                      sequence_len=SEQ_LEN, token_style=token_style,
                                      is_train=True, augment_rate=AUG_RATE,
                                      augment_type=AUG_TYPE)
        test_loader = DataLoader(test_dataset, **data_loader_params)

print('Data was loaded.')

LOG_PATH = model_save_path / 'logs' / f"{args.model_name}_logs.txt"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

WEIGHTS_SAVE_DIR = model_save_path / 'weights'
WEIGHTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def train_epoch(model: nn.Module,
                loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer
                ) -> Tuple[float, float]:
    """
    Train single epoch

    Args:
        model (nn.Module): model instance
        loader (DataLoader): data loader
        criterion (nn.Module): criterion
        optimizer (torch.optim.Optimizer): optimizer

    Returns:
        tuple[float, float]: train_loss, train_accuracy
    """
    train_loss = 0.0
    train_iteration = 0
    correct = 0.
    total = 0.

    model.train()
    for x, y, att, y_mask in tqdm(loader, desc='train'):
        x = x.to(DEVICE)
        y = y.view(-1).to(DEVICE)
        att = att.to(DEVICE)
        y_mask = y_mask.view(-1).to(DEVICE)

        y_predict = model(x, att)

        y_predict = y_predict.view(-1, y_predict.shape[2])
        loss = criterion(y_predict, y)

        y_predict = torch.argmax(y_predict, dim=1).view(-1)
        correct += torch.sum(y_mask * (y_predict == y).long()).item()

        optimizer.zero_grad()
        train_loss += loss.item()
        train_iteration += 1
        loss.backward()

        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

        optimizer.step()
        total += torch.sum(y_mask.view(-1)).item()

    train_loss /= train_iteration
    train_accuracy = correct / total

    return train_loss, train_accuracy


def validate_epoch(model: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module) -> Tuple[float, float, float, float, float]:
    """
    Validate sinlge epoch

    Args:
        model (nn.Module): model instance
        loader (DataLoader): data loader (should be different from train loader)
        criterion (nn.Module): criterion

    Returns:
        tuple[float, float, float, float, float]: validation_loss, validation_accuracy, f1, precision, recall
    """
    num_iteration = 0
    correct = 0.
    total = 0.
    val_loss = 0.0

    # +1 for overall result
    tp = np.zeros(1 + len(TARGETS), dtype=np.int64)
    fp = np.zeros(1 + len(TARGETS), dtype=np.int64)
    fn = np.zeros(1 + len(TARGETS), dtype=np.int64)
    cm = np.zeros((len(TARGETS), len(TARGETS)), dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for x, y, att, y_mask in tqdm(loader, desc='eval'):
            x = x.to(DEVICE)
            y = y.view(-1).to(DEVICE)
            att = att.to(DEVICE)
            y_mask = y_mask.view(-1).to(DEVICE)

            y_predict = model(x, att)
            y_predict = y_predict.view(-1, y_predict.shape[2])

            loss = criterion(y_predict, y)
            val_loss += loss.item()

            y_predict = torch.argmax(y_predict, dim=1).view(-1)

            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()

            for i in range(y.shape[0]):
                if y_mask[i] == 0:
                    # we can ignore this because we know there won't be
                    # any punctuation in this position since we created
                    # this position due to padding or sub-word tokenization
                    continue
                cor = y[i]
                prd = y_predict[i]
                if cor == prd:
                    tp[cor] += 1
                else:
                    fn[cor] += 1
                    fp[prd] += 1
                cm[cor][prd] += 1

    # ignore first index which is for no punctuation
    tp[-1] = np.sum(tp[1:])
    fp[-1] = np.sum(fp[1:])
    fn[-1] = np.sum(fn[1:])

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    val_loss = val_loss / num_iteration
    val_acc = correct / total

    log_test_metrics(LOG_PATH, precision, recall, f1, val_acc, cm)

    non_O_keys = " + ".join(list(TARGETS)[1:])
    targets = list(TARGETS) + [non_O_keys]
    for i, target in enumerate(targets):
        log_target_test_metrics(LOG_PATH, target, precision[i], recall[i], f1[i])

    return val_loss, val_acc, f1, precision, recall


def calc_accuracy_metrics(model: nn.Module,
                          loader: DataLoader
                          ) -> Tuple[ndarray, ndarray, ndarray, float, ndarray]:
    """
    Calculate different accuracy metrics

    Args:
        model (nn.Module): model instance
        loader (DataLoader): data loader

    Returns:
        tuple[ndarray, ndarray, ndarray, float, ndarray]: precision, recall, \
            f1_score, accuracy, confusion_matrx
    """
    num_iteration = 0
    correct = 0
    total = 0

    model.eval()

    # +1 for overall result
    tp = np.zeros(1 + len(TARGETS), dtype=np.int64)
    fp = np.zeros(1 + len(TARGETS), dtype=np.int64)
    fn = np.zeros(1 + len(TARGETS), dtype=np.int64)
    cm = np.zeros((len(TARGETS), len(TARGETS)), dtype=np.int64)

    with torch.no_grad():
        for x, y, att, y_mask in tqdm(loader, desc='test'):
            x = x.to(DEVICE)
            y = y.view(-1).to(DEVICE)
            att = att.to(DEVICE)
            y_mask = y_mask.view(-1).to(DEVICE)

            y_predict = model(x, att)
            y_predict = y_predict.view(-1, y_predict.shape[2])
            y_predict = torch.argmax(y_predict, dim=1).view(-1)

            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += int(torch.sum(y_mask * (y_predict == y).long()).item())
            total += int(torch.sum(y_mask).item())
            for i in range(y.shape[0]):
                if y_mask[i] == 0:
                    # we can ignore this because we know there won't be
                    # any punctuation in this position since we created
                    # this position due to padding or sub-word tokenization
                    continue
                cor = y[i]
                prd = y_predict[i]
                if cor == prd:
                    tp[cor] += 1
                else:
                    fn[cor] += 1
                    fp[prd] += 1
                cm[cor][prd] += 1

    # ignore first index which is for no punctuation
    tp[-1] = np.sum(tp[1:])
    fp[-1] = np.sum(fp[1:])
    fn[-1] = np.sum(fn[1:])

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = correct / total

    return precision, recall, f1, accuracy, cm


def train() -> None:
    """Train: global function"""
    if not args.resume:
        best_val_acc = 0.0
        epochs = range(args.epoch)
    else:
        last_epoch, best_val_acc = get_last_epoch_params(orig_model_dir / 'weights')
        epochs = range(last_epoch + 1, last_epoch + 1 + args.epoch)

    # TODO: continue logging, remove print
    with experiment.record(name=MODEL_SAVE_NAME, exp_conf=args.__dict__) if args.labml else ExitStack():
        for epoch in epochs:
            train_loss, train_acc = train_epoch(MODEL, train_loader, CRITERION, OPTIMIZER)
            log_train_epoch(LOG_PATH, epoch, train_loss, train_acc)

            val_loss, val_acc, f1, precision, recall = validate_epoch(MODEL, val_loader, CRITERION)
            log_val_epoch(LOG_PATH, epoch, val_loss, val_acc)

            if args.labml:
                tracker.save(epoch, {'train_loss': train_loss,
                                     'train_accuracy': train_acc,
                                     'val_loss': val_loss,
                                     'val_accuracy': val_acc,
                                     'f1': f1,
                                     'precision': precision,
                                     'recall': recall})

            if args.store_every_weight:
                save_weights(MODEL, WEIGHTS_SAVE_DIR, epoch, val_acc)
            elif args.store_best_weights and (val_acc > best_val_acc):
                best_val_acc = val_acc
                save_weights(MODEL, WEIGHTS_SAVE_DIR, epoch, val_acc)

    log_text(LOG_PATH, f"Best validation Acc: {best_val_acc}")


def test() -> None:
    """Test: global function"""
    # precision, recall, f1, accuracy, cm = calc_accuracy_metrics(MODEL, test_loader)
    precision, recall, f1, accuracy, cm = calc_accuracy_metrics(MODEL, val_loader)
    log_test_metrics(LOG_PATH, precision, recall, f1, accuracy, cm)

    non_O_keys = " + ".join(list(TARGETS)[1:])
    targets = list(TARGETS) + [non_O_keys]
    for i, target in enumerate(targets):
        log_target_test_metrics(LOG_PATH, target, precision[i], recall[i], f1[i])


if __name__ == '__main__':
    export_params(args, model_save_path)
    log_args(LOG_PATH, args)
    train()
    if args.test_data:
        test()
