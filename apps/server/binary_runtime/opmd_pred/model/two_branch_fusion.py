"""Research-only photo/table fusion, independent of deployed five-modality APIs."""
import torch
from torch import nn
from torch.nn import functional as F


class TwoBranchFusion(nn.Module):
    def __init__(self, feature_dim, table_dim, num_classes, family='transformer', dimension=128, state_size=64):
        super().__init__()
        self.family = family
        self.num_classes = num_classes
        if family == 'image_only':
            self.head = nn.Linear(feature_dim, num_classes)
        elif family == 'transformer':
            self.image_project = nn.Sequential(nn.Linear(feature_dim, dimension), nn.LayerNorm(dimension), nn.GELU())
            self.table_project = nn.Sequential(nn.Linear(table_dim, dimension), nn.LayerNorm(dimension), nn.GELU())
            self.missing = nn.Parameter(torch.zeros(2, dimension))
            self.kind = nn.Parameter(torch.randn(2, dimension) * .02)
            self.cls = nn.Parameter(torch.zeros(1, 1, dimension))
            layer = nn.TransformerEncoderLayer(dimension, 4, dimension * 4, dropout=.1,
                                               activation='gelu', batch_first=True, norm_first=True)
            self.fusion = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, num_classes))
        elif family == 'multimodn':
            self.initial_state = nn.Parameter(torch.zeros(state_size))
            def update(n):
                return nn.Sequential(nn.Linear(state_size + n, state_size), nn.GELU(),
                                     nn.LayerNorm(state_size), nn.Linear(state_size, state_size))
            self.table_update = update(table_dim)
            self.image_update = update(feature_dim)
            self.head = nn.Linear(state_size, num_classes)
        else:
            raise ValueError('Unknown research fusion family')

    def forward(self, batch, return_steps=False):
        x, t = batch['image'], batch['table']
        ip, tp = batch['image_present'].bool(), batch['table_present'].bool()
        available = ip if self.family == 'image_only' else ip | tp
        if not available.all():
            raise ValueError('No selected input: reject before model forward')
        # Absent inputs may contain NaN; remove them before applying learned maps.
        x = torch.where(ip[:, None], x, torch.zeros_like(x))
        t = torch.where(tp[:, None], t, torch.zeros_like(t))
        if self.family == 'image_only':
            return self.head(x)
        if self.family == 'transformer':
            zt = torch.where(tp[:, None], self.table_project(t), self.missing[0]) + self.kind[0]
            zi = torch.where(ip[:, None], self.image_project(x), self.missing[1]) + self.kind[1]
            sequence = torch.cat([self.cls.expand(len(x), -1, -1), torch.stack([zt, zi], 1)], 1)
            return self.head(self.fusion(sequence)[:, 0])
        state = self.initial_state.expand(len(x), -1)
        steps = []
        for value, present, update in [(t, tp, self.table_update), (x, ip, self.image_update)]:
            ids = present.nonzero(as_tuple=True)[0]
            if len(ids):
                next_state = state.clone()
                next_state[ids] = update(torch.cat([state[ids], value[ids]], 1))
                state = next_state
            steps.append((self.head(state), present))
        return (self.head(state), steps) if return_steps else self.head(state)

    def supervised_loss(self, batch, weights):
        y = batch['target']
        if self.family != 'multimodn':
            return F.cross_entropy(self(batch), y, weight=weights)
        _, steps = self(batch, return_steps=True)
        summed = torch.zeros(len(y), device=y.device)
        count = torch.zeros_like(summed)
        for logits, present in steps:
            summed = summed + F.cross_entropy(logits, y, reduction='none') * present
            count = count + present
        per_row = summed / count.clamp_min(1)
        return (per_row * weights[y]).sum() / weights[y].sum()


def drop_branches(batch, probability):
    result = dict(batch)
    mask = torch.stack([batch['table_present'], batch['image_present']], 1).bool()
    kept = mask & (torch.rand(mask.shape, device=mask.device) >= probability)
    empty = ~kept.any(1)
    if empty.any():
        # Restore a randomly selected actually present branch, never a missing one.
        scores = torch.rand(mask.shape, device=mask.device).masked_fill(~mask, -1.)
        restore = scores.argmax(1)
        kept[empty, restore[empty]] = True
    result['table_present'], result['image_present'] = kept[:, 0], kept[:, 1]
    return result
