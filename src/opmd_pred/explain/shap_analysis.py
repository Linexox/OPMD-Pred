# -*- coding: utf-8 -*-
"""表格特征级精确 Shapley 值（12 特征，2^12=4096 联盟全枚举）。

约定：
- 被屏蔽的特征取"背景值"：背景 = 训练折样本的 12 维已 scaling 向量，取期望。
- 模态 present mask 与图像嵌入固定为该样本的真实状态（解释的是
  "给定模态存在性下，各特征取值对预测的贡献"）。
- 因 12 特征可全枚举，直接用 Shapley 公式，无需近似。
"""

from __future__ import annotations

from math import factorial

import numpy as np
import torch

from ..models.cmf_model import MODALITY_ORDER

M_TAB = 12  # 10 exfoliative + 1 methylation + 1 age


def _model_fn(model, sample, tab_batch: np.ndarray, device: str) -> np.ndarray:
    """(N,12) 表格向量 → (N,) 连续预测。present mask / 图像固定。"""
    model.eval()
    natural = torch.from_numpy(sample.natural_missing()).to(device)     # (4,)
    if sample.has_image:
        img = torch.from_numpy(sample.image_eval).to(device)
        img = img.unsqueeze(0).expand(tab_batch.shape[0], -1)
    else:
        img = torch.zeros(tab_batch.shape[0], model.cfg.image.embed_dim, device=device)
    x = {
        "image": img,
        "exfoliative": torch.from_numpy(tab_batch[:, 0:10]).to(device),
        "methylation": torch.from_numpy(tab_batch[:, 10:11]).to(device),
        "basic": torch.from_numpy(tab_batch[:, 11:12]).to(device),
    }
    missing = natural.unsqueeze(0).expand(tab_batch.shape[0], -1)
    with torch.no_grad():
        pred = model.predict_continuous(x, missing)
    return pred.cpu().numpy()


def tabular_shapley(model, sample, background: np.ndarray, device: str,
                    chunk: int = 2048) -> np.ndarray:
    """返回 (12,) 精确 Shapley 值。background: (n_bg, 12) 训练折样本。

    12 个特征 → 2^12 = 4096 个联盟全枚举（无需 KernelExplainer 近似）。
    v(S) = E_b f(z⊙1_S + b⊙(1-1_S))，b 取训练折样本。
    """
    n_bg = background.shape[0]
    assert background.shape[1] == M_TAB

    z = sample.tab                                     # (12,)
    v = np.zeros(2 ** M_TAB)
    for s in range(2 ** M_TAB):
        keep = np.array([(s >> i) & 1 == 1 for i in range(M_TAB)], dtype=bool)
        rows = np.where(keep[None, :], z[None, :], background)   # (n_bg, 12)
        outs = [_model_fn(model, sample, rows[i:i + chunk], device)
                for i in range(0, n_bg, chunk)]
        v[s] = float(np.mean(np.concatenate(outs)))

    phi = np.zeros(M_TAB)
    for i in range(M_TAB):
        rest = [j for j in range(M_TAB) if j != i]
        for S_int in range(2 ** (M_TAB - 1)):
            S = 0
            for k, j in enumerate(rest):
                if (S_int >> k) & 1:
                    S |= 1 << j
            w = factorial(bin(S).count("1")) * factorial(M_TAB - 1 - bin(S).count("1")) / factorial(M_TAB)
            phi[i] += w * (v[S | (1 << i)] - v[S])
    return phi


def subsample_background(samples, max_n: int = 32, seed: int = 0) -> np.ndarray:
    """训练集太大时抽背景（等距抽取，保持分布代表性）。"""
    tab = np.stack([s.tab for s in samples])
    if len(tab) <= max_n:
        return tab.astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tab), size=max_n, replace=False)
    return tab[idx].astype(np.float32)


def modality_group_shapley(phi: np.ndarray) -> dict[str, float]:
    """把 12 个特征 |φ| 按模态聚合（与 InterSHAP 模态级互为对照）。"""
    return {
        "exfoliative": float(np.abs(phi[0:10]).sum()),
        "methylation": float(np.abs(phi[10:11]).sum()),
        "basic": float(np.abs(phi[11:12]).sum()),
    }
