import torch
import torch.nn.functional as F


def ordinal_loss(logits, labels):
    thresholds = torch.stack([labels > 0, labels > 1], dim=1).float()
    return F.binary_cross_entropy_with_logits(logits, thresholds)


def corn_loss(logits, labels):
    """Two conditional tasks; the second is supervised only where the true label exceeds zero."""
    first = F.binary_cross_entropy_with_logits(logits[:, 0], (labels > 0).float(), reduction="sum")
    eligible = labels > 0
    second = logits.sum() * 0.0
    if eligible.any():
        second = F.binary_cross_entropy_with_logits(logits[eligible, 1], (labels[eligible] > 1).float(), reduction="sum")
    return (first + second) / (labels.numel() + eligible.sum())


def classification_loss(logits, labels, head="softmax"):
    if head == "ordinal":
        return ordinal_loss(logits, labels)
    if head == "corn":
        return corn_loss(logits, labels)
    return F.cross_entropy(logits, labels)


def cumulative_distance_loss(probabilities, labels):
    """Squared cumulative distance to the ordered one-hot target, summed across two cutpoints."""
    cumulative = torch.stack([probabilities[:, 1:].sum(dim=-1), probabilities[:, 2]], dim=-1)
    target = torch.stack([labels > 0, labels > 1], dim=-1).to(probabilities.dtype)
    return (cumulative - target).square().sum(dim=-1).mean()


def info_nce_loss(first, second, temperature=0.1):
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    logits = first @ second.T / temperature
    targets = torch.arange(first.size(0), device=first.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
