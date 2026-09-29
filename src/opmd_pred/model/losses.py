import torch
import torch.nn.functional as F


def ordinal_loss(logits, labels, sample_weight=None):
    thresholds = torch.stack([labels > 0, labels > 1], dim=1).float()
    loss = F.binary_cross_entropy_with_logits(logits, thresholds, reduction="none")
    loss = loss.mean(dim=1)
    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()


def info_nce_loss(first, second, temperature=0.1):
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    logits = first @ second.T / temperature
    targets = torch.arange(first.size(0), device=first.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
