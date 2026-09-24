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
