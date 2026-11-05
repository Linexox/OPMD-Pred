"""Gating over complete frozen photo/table classifiers, never over table branches."""
import torch
from torch import nn


@torch.no_grad()
def full_model_features(model, batch):
    """Return the actual Transformer input H0 and final two-class probabilities."""
    if model.family != 'transformer' or model.num_classes != 2 or model.training:
        raise ValueError('Experts must be frozen binary Transformer models in eval mode')
    captured = []
    handle = model.fusion.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        probabilities = model(batch).softmax(1)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError('Expected exactly one fusion call')
    return captured[0].detach(), probabilities.detach()


class FullModelRouter(nn.Module):
    """One weight per complete expert; zero initialization reproduces averaging."""
    def __init__(self, experts=5, dimension=128, mode='dynamic'):
        super().__init__()
        self.experts, self.dimension, self.mode = experts, dimension, mode
        if mode == 'dynamic':
            self.router = nn.Linear(experts * 3 * dimension, experts)
            nn.init.zeros_(self.router.weight)
            nn.init.zeros_(self.router.bias)
        elif mode == 'constant':
            self.logits = nn.Parameter(torch.zeros(experts))
        elif mode != 'mean':
            raise ValueError('Unknown ensemble mode')

    def forward(self, initial_sequences, expert_probabilities):
        n = len(expert_probabilities)
        if initial_sequences.shape != (n, self.experts, 3, self.dimension):
            raise ValueError('Expected one complete initial sequence per expert')
        if expert_probabilities.shape != (n, self.experts, 2):
            raise ValueError('Expected final binary probabilities per complete expert')
        if self.mode == 'dynamic':
            logits = self.router(initial_sequences.flatten(1))
        elif self.mode == 'constant':
            logits = self.logits.expand(n, -1)
        else:
            logits = initial_sequences.new_zeros((n, self.experts))
        weights = logits.softmax(1)
        return (expert_probabilities * weights[:, :, None]).sum(1), weights
