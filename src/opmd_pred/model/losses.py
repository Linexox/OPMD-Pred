import torch
import torch.nn.functional as F


def ordinal_loss(logits, labels):
    thresholds = torch.stack([labels > 0, labels > 1], dim=1).float()
    return F.binary_cross_entropy_with_logits(logits, thresholds)


def observed_subset_consistency(teacher_probabilities, student_probabilities, kind="cdf_mse"):
    """Teacher uses actual observed inputs; only the target distribution is detached."""
    teacher = teacher_probabilities.detach()
    if kind == "forward_kl":
        return F.kl_div(student_probabilities.clamp_min(1e-8).log(), teacher, reduction="batchmean")
    if kind == "cdf_mse":
        teacher_q = torch.stack([teacher[:, 1:].sum(dim=1), teacher[:, 2]], dim=1)
        student_q = torch.stack([student_probabilities[:, 1:].sum(dim=1), student_probabilities[:, 2]], dim=1)
        return (student_q - teacher_q).square().sum(dim=1).mean()
    raise ValueError(f"Unsupported consistency type: {kind}")


def info_nce_loss(first, second, temperature=0.1):
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    logits = first @ second.T / temperature
    targets = torch.arange(first.size(0), device=first.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
