import random
from pathlib import Path

import torch
from loguru import logger


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_logger(output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(lambda message: print(message, end=""))
    logger.add(output_dir / "train.log")
    return logger


def device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"


def classification_metrics(logits, labels):
    predictions = logits.argmax(dim=1)
    return {
        "accuracy": (predictions == labels).float().mean().item(),
        "mae": (predictions - labels).abs().float().mean().item(),
    }


def multiclass_metrics(predictions, labels, class_count=4):
    confusion = torch.bincount(
        labels * class_count + predictions,
        minlength=class_count * class_count,
    ).reshape(class_count, class_count).float()
    support = confusion.sum(dim=1)
    recall = confusion.diag() / support.clamp_min(1)
    precision = confusion.diag() / confusion.sum(dim=0).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "support": support.to(torch.long).tolist(),
        "recall": recall.tolist(),
        "macro_f1": f1.mean().item(),
        "balanced_accuracy": recall[support > 0].mean().item(),
        "confusion_matrix": confusion.to(torch.long).tolist(),
    }
