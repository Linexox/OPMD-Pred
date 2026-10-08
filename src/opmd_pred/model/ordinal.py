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
    """Legacy posterior-median decision; new SYSU CLI uses probability argmax."""
    return (logits.sigmoid() > 0.5).sum(dim=-1)


def model_probabilities(logits, head="ordinal"):
    if head == "ordinal":
        return ordinal_probabilities(logits)
    if head == "corn":
        conditional = logits.sigmoid()
        first = conditional[..., 0]
        second = first * conditional[..., 1]
        return torch.stack([1 - first, first - second, second], dim=-1)
    return logits.softmax(dim=-1)


def probability_decision(probabilities, cost_matrix=None):
    if cost_matrix is None:
        return probabilities.argmax(dim=-1)
    return (probabilities @ cost_matrix).argmin(dim=-1)
