"""Training-only controls for the existing complete-model ensemble."""
import torch
from torch import nn


def importance_balance_loss(weights):
    """Squared CV of mean soft weights; balance records collectively, not each row."""
    if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[1] < 2:
        raise ValueError('Expected nonempty record-by-expert weights')
    experts = weights.shape[1]
    return experts * (weights.mean(0) - 1 / experts).square().sum()


def set_transformer_dropout(model, probability):
    """Change both attention and activation/residual dropout without changing weights."""
    if model.family != 'transformer' or not 0 <= probability < 1:
        raise ValueError('Expected Transformer and dropout in [0,1)')
    for module in model.fusion.modules():
        if isinstance(module, nn.Dropout):
            module.p = probability
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = probability
    return model
