import json
import re
from argparse import Namespace
from pathlib import Path
from typing import Tuple

from neuro_comma.model import CorrectionModel


def get_model_save_path(model_dir: Path, args: Namespace) -> Path:
    """
    Get path, where to save a model. It also looking for already existed directories \
        and generate new name, to not override.

    Args:
        model_dir (Path): parent directory, where model will be stored
        args (Namespace): args from argparse

    Returns:
        Path: directory, where model will be stored
    """
    def generate_new_save_path(path: Path) -> Path:
        version = 1
        new_path = path.with_stem(f"{path.stem}^{version}")
        while new_path.is_dir():
            version += 1
            new_path = path.with_stem(f"{path.stem}^{version}")
        return new_path

    if args.fine_tune:
        model_save_path = model_dir / f"{args.model_name}_ft"
        if model_save_path.is_dir():
            model_save_path = generate_new_save_path(model_save_path)
        return model_save_path

    model_save_path = model_dir / args.model_name
    if (not args.resume) and model_save_path.is_dir():
        model_save_path = generate_new_save_path(model_save_path)
    return model_save_path


def get_last_pretrained_weight_path(models_dir: Path) -> Path:
    """Get last pretrained weight path

    Args:
        models_dir (Path): directory, where model is stored

    Raises:
        FileNotFoundError: if no weights for model

    Returns:
        Path: weight's path
    """
    weights_dir = models_dir / 'weights'
    weights = list(weights_dir.glob('**/*.pt'))
    if not weights:
        raise FileNotFoundError(f"No weights here: {weights_dir}")
    last_weight = sorted(weights)[-1]
    return last_weight


def export_params(args: Namespace, model_dir: Path) -> None:
    """
    Export args to json

    Args:
        args (Namespace): args from argparse
        model_dir (Path): model directory path
    """
    params = args.__dict__
    file_name = model_dir / 'params.json'
    with open(file_name, 'w') as f:
        json.dump(params, f)


def load_params(model_dir: Path) -> dict:
    """
    Load params from json

    Args:
        model_dir (Path): model directory path

    Returns:
        dict: dictionary of params
    """
    file_name = model_dir / 'params.json'
    with open(file_name, 'r') as f:
        params = json.load(f)
    return params


def save_weights(model: CorrectionModel,
                 weights_dir: Path,
                 epoch: int,
                 accuracy: float) -> None:
    """
    Save weights with epoch number and accuracy

    Args:
        model (CorrectionModel): model instance
        weights_dir (Path): path to store weight
        epoch (int): epoch number
        accuracy (float): accuracy
    """
    acc = str(accuracy)[2:6]
    save_path = weights_dir / f"weights_ep{epoch}_{acc}.pt"
    model.save(save_path)


def get_last_epoch_params(weights_dir: Path) -> Tuple[int, float]:
    """
    Get last epoch number and last best accuracy

    Args:
        weights_dir (Path): directory where wheights are stored

    Returns:
        tuple[int, float]: epoch number, best accuracy
    """
    weights = list(weights_dir.glob('**/*.pt'))
    last_weight = str(sorted(weights)[-1])
    if (match := re.search(r'_ep(\d+)_(\d+)\.pt', last_weight)):
        epoch = int(match.group(1))
        best_acc = float('0.' + match.group(2))
        return epoch, best_acc
    return 0, 0.0


# if __name__ == "__main__":
    # model_dir = Path('/media/sviperm/9740514d-d8c8-4f3e-afee-16ce6923340c2/sviperm/Documents/voicetextassistant.ai/contextual-mistakes/models/debug-model')
    # d = load_params(model_dir)
    # print(d)

    # weights_dir = Path("/media/sviperm/9740514d-d8c8-4f3e-afee-16ce6923340c2/sviperm/Documents/voicetextassistant.ai/contextual-mistakes/models/debug-model_ft/weights")
    # e = get_last_epoch_params(weights_dir)
    # print(e)
