# -*- coding: utf-8 -*-
"""损失函数：有监督 sigmoid 对比损失（Gu 2025）+ CORAL 序数回归损失。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _one_direction(za: torch.Tensor, zb: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    """anchor=za 的一侧。正样本：自身(跨模态对) + 同类；负样本：异类。"""
    sim = za @ zb.T / tau                      # (B, B)
    same = y[:, None] == y[None, :]            # (B, B)
    n = za.shape[0]
    losses = []
    for i in range(n):
        pos = same[i].clone()
        pos[i] = True                          # 对角（同一样本跨模态）恒为正
        neg = ~same[i]                         # i 行对角为 True，不会进负样本
        lp = -F.logsigmoid(sim[i][pos]).mean()
        ln = -F.logsigmoid(-sim[i][neg]).mean() if neg.any() else za.new_zeros(())
        losses.append(lp + ln)
    return torch.stack(losses).mean()


def supervised_sigmoid_contrastive(
    za: torch.Tensor, zb: torch.Tensor, y: torch.Tensor, tau: float = 0.5
) -> torch.Tensor:
    """双向对称的 sigmoid 型有监督对比损失。

    za/zb: (B, D) 已 L2 归一化的两个视图（两个模态，或 模态×融合）
    y:     (B,)   序数标签（同类视为正样本对）
    """
    if za.shape[0] < 2:
        return za.new_zeros(())
    return (_one_direction(za, zb, y, tau) + _one_direction(zb, za, y, tau)) / 2


def coral_loss(logits: torch.Tensor, y: torch.Tensor, n_classes: int = 4) -> torch.Tensor:
    """CORAL 累积链接损失。logits: (B, K-1)；目标 y_k = 1[y > k]。"""
    k = torch.arange(n_classes - 1, device=logits.device, dtype=y.dtype)
    targets = (y[:, None] > k[None, :]).float()
    return F.binary_cross_entropy_with_logits(logits, targets)
