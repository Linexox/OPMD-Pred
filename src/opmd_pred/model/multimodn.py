"""Small sequential modality updates, adapted from MultiModN for SYSU grading."""
import torch
from torch import nn
from torch.nn import functional as F


class MultiModNClassifier(nn.Module):
    """Update only observed rows; supervise each row's observed update steps."""

    names = ("history", "image", "tct", "dna", "methylation")

    def __init__(self, feature_dim=512, state_size=64, tct_categories=8):
        super().__init__()
        self.feature_dim, self.state_size = feature_dim, state_size
        self.initial_state = nn.Parameter(torch.zeros(state_size))
        self.tct_embedding = nn.Embedding(max(2, tct_categories + 1), 16)
        sizes = {"history": 18, "image": feature_dim, "tct": 16, "dna": 4, "methylation": 2}
        self.updates = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(state_size + sizes[name], state_size), nn.GELU(),
            nn.LayerNorm(state_size), nn.Linear(state_size, state_size)) for name in self.names})
        self.classifier = nn.Linear(state_size, 3)

    def inputs(self, batch):
        values, present = {}, {}
        for name in self.names:
            if name == "image":
                values[name], present[name] = batch["image_features"], batch["image_present"].bool()
            elif name == "tct":
                values[name], present[name] = self.tct_embedding(batch["tct"]), batch["tct_present"].bool()
            else:
                mask = batch[f"{name}_mask"].bool()
                values[name] = torch.cat((batch[name].masked_fill(~mask, 0), mask.float()), dim=1)
                present[name] = mask.any(1)
        return values, present

    def steps(self, batch):
        values, present = self.inputs(batch)
        count = torch.stack([present[n] for n in self.names], 1).sum(1)
        if not bool(count.gt(0).all()):
            raise ValueError("No available input: exclude empty records before model inference")
        state = self.initial_state.expand(len(count), -1)
        predictions, trace = [], []
        for name in self.names:
            rows = present[name].nonzero().flatten()
            before = state
            if rows.numel():
                update = self.updates[name](torch.cat((state[rows], values[name][rows]), 1))
                state = state.index_copy(0, rows, update)
                predictions.append((rows, self.classifier(state[rows])))
            trace.append((name, before, state, rows))
        return self.classifier(state), predictions, count, trace

    def forward(self, batch):
        return self.steps(batch)[0]

    def supervised_loss(self, batch, weights):
        _, predictions, count, _ = self.steps(batch)
        labels = batch["label"]
        accumulated = torch.zeros(len(labels), device=labels.device)
        for rows, logits in predictions:
            loss = F.cross_entropy(logits, labels[rows], reduction="none")
            accumulated = accumulated.index_add(0, rows, loss)
        # Same class-weighted batch normalization as PyTorch weighted CE, while
        # each patient's available updates share one total sample weight.
        row_weights = weights[labels]
        return ((accumulated / count) * row_weights).sum() / row_weights.sum()
