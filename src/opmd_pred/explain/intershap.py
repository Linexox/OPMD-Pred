# -*- coding: utf-8 -*-
"""InterSHAP（Wenderoth 2025, AAAI）的 4 模态精确实现。

核心优势（本项目的差异化卖点）：屏蔽一个模态 = 把它的输入替换为模型自己的
learnable missing embedding（训练时模态 dropout 用的同一个 token），
而不是原文的"换成另一条样本"，因此 masking 不引入分布外扰动。

定义（对照 Wenderoth 原文 Eq.）：
  v(S) = 模型在"仅保留 S 中（且自然存在）的模态"下的连续输出
  φ_i  = 标准 Shapley 值，权重 |S|!(n-|S|-1)!/n!
  Φ_ij = 成对 Shapley 交互指数(SII)，权重 |S|!(n-|S|-2)!/(n-1)!
  Φ_ii = φ_i − Σ_{j≠i} Φ_ij（模态自身贡献，原文的定义）
  local InterSHAP  = Σ_{i≠j}|Φ_ij| / Σ_{i,j}|Φ_ij|
  global InterSHAP = 先对 |Φ_ij| 跨样本取均值再算同一比值（防正负抵消虚高）

注意：φ 与 Φ 必须用上面两套各自的权重，否则 Φ_ii = φ_i − ΣΦ_ij 会失效
（两套权重混用会使对角与非对角差 n=4 时的 16 倍，见 scripts/smoke_pipeline.py 的自检）。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import factorial

import numpy as np
import torch

from ..models.cmf_model import MODALITY_ORDER

N_MOD = len(MODALITY_ORDER)


@dataclass
class InterShapResult:
    phi_matrix: np.ndarray            # (B, 4, 4)，含对角 Φ_ii
    intershap_local: np.ndarray       # (B,)
    modality_shapley: np.ndarray      # (B, 4)，经典 Shapley 主效应 φ_i（供参考）
    complete_case_mask: np.ndarray    # (B,) bool，四模态齐全的样本


# ---------------------------------------------------------------------------
# 权重：单玩家 Shapley 与成对 SII 用不同分母，不可混用
# ---------------------------------------------------------------------------
def shapley_weight(s: int, n: int = N_MOD) -> float:
    """单玩家 Shapley 权重：|S|!(n-|S|-1)!/n!。"""
    return factorial(s) * factorial(n - s - 1) / factorial(n)


def sii_weight(s: int, n: int = N_MOD, order: int = 2) -> float:
    """|T|=order 的 Shapley 交互指数权重：|S|!(n-|S|-order)!/(n-order+1)!。"""
    return factorial(s) * factorial(n - s - order) / factorial(n - order + 1)


def _bitset_from_int(S_int: int, players: list[int]) -> int:
    """把 S_int 的第 k 位映射到 players[k] 对应的 bitmask。"""
    S = 0
    for k, j in enumerate(players):
        if (S_int >> k) & 1:
            S |= 1 << j
    return S


def _popcount(x: int) -> int:
    return bin(x).count("1")


# ---------------------------------------------------------------------------
# 联盟值：屏蔽 = 换成 missing token
# ---------------------------------------------------------------------------
def _coalition_values(model, sample, device: str) -> np.ndarray:
    """单样本 16 个联盟的 v(S)。v = 连续预测（MSE 输出 / CORAL 期望值）。"""
    natural = torch.from_numpy(sample.natural_missing()).to(device)     # (4,)
    img = (torch.from_numpy(sample.image_eval).to(device)
           if sample.has_image else torch.zeros(model.cfg.image.embed_dim, device=device))
    tab = torch.from_numpy(sample.tab).to(device)
    x = {
        "image": img.unsqueeze(0),
        "exfoliative": tab[0:10].unsqueeze(0),
        "methylation": tab[10:11].unsqueeze(0),
        "basic": tab[11:12].unsqueeze(0),
    }
    vals = np.zeros(2 ** N_MOD, dtype=np.float64)
    with torch.no_grad():
        for s in range(2 ** N_MOD):
            keep = torch.tensor(
                [(s >> i) & 1 == 1 for i in range(N_MOD)], dtype=torch.bool, device=device)
            missing = natural | ~keep
            vals[s] = float(model.predict_continuous(x, missing.unsqueeze(0)).item())
    return vals


# ---------------------------------------------------------------------------
# 由 16 个联盟值精确算 4×4 SII 矩阵
# ---------------------------------------------------------------------------
def shapley_values(v: np.ndarray, n: int = N_MOD) -> np.ndarray:
    """标准 Shapley 值 φ_i（(n,)）。v: 长度 2^n 的联盟值数组。"""
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for S_int in range(2 ** (n - 1)):
            S = _bitset_from_int(S_int, others)
            w = shapley_weight(_popcount(S), n)
            phi[i] += w * (v[S | (1 << i)] - v[S])
    return phi


def sii_matrix(v: np.ndarray, n: int = N_MOD) -> np.ndarray:
    """4×4 SII 交互矩阵：非对角 Φ_ij（成对 SII），对角 Φ_ii = φ_i − Σ_{j≠i} Φ_ij。"""
    Phi = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        others = [k for k in range(n) if k not in (i, j)]
        for S_int in range(2 ** (n - 2)):
            S = _bitset_from_int(S_int, others)
            w = sii_weight(_popcount(S), n, order=2)
            delta = v[S | (1 << i) | (1 << j)] - v[S | (1 << i)] - v[S | (1 << j)] + v[S]
            val = w * delta
            Phi[i, j] += val
            Phi[j, i] += val
    phi = shapley_values(v, n)
    # Φ_ii = φ_i − Σ_{j≠i} Φ_ij（Wenderoth 原文定义）
    for i in range(n):
        Phi[i, i] = phi[i] - sum(Phi[i, j] for j in range(n) if j != i)
    return Phi


def _local_score(Phi: np.ndarray) -> float:
    """InterSHAP_i = Σ_{i≠j}|Φ_ij| / Σ_{i,j}|Φ_ij|。"""
    total = float(np.abs(Phi).sum())
    if total < 1e-12:
        return 0.0
    off = total - float(np.abs(np.diag(Phi)).sum())
    return off / total


def intershap(model, samples, device: str, verbose: bool = False) -> InterShapResult:
    """对一批样本逐个计算 local InterSHAP。每个样本 16 次前向。"""
    model.eval()
    phis, locals_, shapleys, cc = [], [], [], []
    for si, s in enumerate(samples):
        v = _coalition_values(model, s, device)
        Phi = sii_matrix(v)
        phis.append(Phi)
        locals_.append(_local_score(Phi))
        shapleys.append(shapley_values(v))
        cc.append(bool(not s.natural_missing().any()))
        if verbose and (si + 1) % 10 == 0:
            print(f"    [intershap] {si+1}/{len(samples)}")
    return InterShapResult(
        phi_matrix=np.stack(phis),
        intershap_local=np.array(locals_),
        modality_shapley=np.stack(shapleys),
        complete_case_mask=np.array(cc),
    )


def global_intershap(res: InterShapResult, complete_case_only: bool = False) -> tuple[float, np.ndarray]:
    """返回 (global InterSHAP 分数, 4×4 平均|Φ|矩阵)。"""
    mask = res.complete_case_mask if complete_case_only else np.ones(len(res.phi_matrix), dtype=bool)
    if mask.sum() == 0:
        return float("nan"), np.full((N_MOD, N_MOD), np.nan)
    mean_abs = np.abs(res.phi_matrix[mask]).mean(axis=0)
    return _local_score(mean_abs), mean_abs
