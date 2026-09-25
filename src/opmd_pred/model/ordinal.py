import torch


def ordinal_probabilities(logits):
    """Convert cumulative ordinal logits to probabilities for classes 0, 1, and 2."""
    cumulative = logits.sigmoid()
    return torch.stack(
        [1.0 - cumulative[..., 0], cumulative[..., 0] - cumulative[..., 1], cumulative[..., 1]],
        dim=-1,
    )


def ordinal_expected_severity(logits):
    return logits.sigmoid().sum(dim=-1)


def ordinal_prediction(logits):
    return (logits.sigmoid() > 0.5).sum(dim=-1)
