# -*- coding: utf-8 -*-
"""阶段B（对比预训练）与阶段C（任务训练）的训练循环。

设计要点（对应 Gu 2025）：
- 阶段B：有监督 sigmoid 对比损失，10 项 = 模态两两 6 项 + 模态×融合 4 项；
  只用自然存在的模态参与对应项；全批量（≤56 样本）。
- 阶段C：全组合同步 modality dropout —— 2^4-1=15 种非空组合全部显式监督，
  每个组合下缺失模态用 learnable token 替换；任务损失 MSE(主) / CORAL(消融)。
- 早停：验证集 QWK（并列时看 MAE），保存最优权重。
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from torch import nn

from ..config import TrainConfig
from ..models.cmf_model import CMFModel, MODALITY_IDX, MODALITY_ORDER
from ..models.losses import coral_loss, supervised_sigmoid_contrastive
from .dataset import FoldData, collate
from .metrics import ordinal_metrics

# 15 种非空模态组合（True=保留）
ALL_COMBOS = [
    torch.tensor(c, dtype=torch.bool)
    for c in itertools.product([False, True], repeat=len(MODALITY_ORDER))
][1:]


def _contrastive_terms(out: dict, y: torch.Tensor, missing: torch.Tensor, tau: float) -> tuple[torch.Tensor, int]:
    """10 项对比损失的均值。missing: (B,4) bool。"""
    present = ~missing
    terms: list[torch.Tensor] = []
    for a, b in itertools.combinations(MODALITY_ORDER, 2):
        both = present[:, MODALITY_IDX[a]] & present[:, MODALITY_IDX[b]]
        if both.sum() >= 2:
            terms.append(supervised_sigmoid_contrastive(
                out["p"][a][both], out["p"][b][both], y[both], tau))
    for a in MODALITY_ORDER:
        m = present[:, MODALITY_IDX[a]]
        if m.sum() >= 2:
            terms.append(supervised_sigmoid_contrastive(
                out["p"][a][m], out["z_f"][m], y[m], tau))
    if not terms:
        return y.new_zeros(()), 0
    return torch.stack(terms).mean(), len(terms)


def pretrain_fold(
    model: CMFModel, data: FoldData, cfg: TrainConfig, device: str,
    rng: np.random.Generator, log_every: int = 20,
) -> list[float]:
    """阶段B：只在 fold.train 上做对比预训练。返回逐 epoch 损失。"""
    model.to(device).train()
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.pretrain_lr, weight_decay=cfg.weight_decay,
    )
    x, y, missing = collate(data.train, train=True, rng=rng)
    x = {k: v.to(device) for k, v in x.items()}
    y, missing = y.to(device), missing.to(device)

    history: list[float] = []
    for ep in range(cfg.pretrain_epochs):
        opt.zero_grad()
        out = model(x, missing)
        loss, n_terms = _contrastive_terms(out, y, missing, cfg.temperature)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        if log_every and (ep + 1) % log_every == 0:
            print(f"    [pretrain ep{ep+1}] contrastive={loss.item():.4f} ({n_terms} terms)")
    return history


def _task_loss(model: CMFModel, x, y, missing: torch.Tensor, head: str) -> torch.Tensor:
    out = model(x, missing)
    if head == "mse":
        return nn.functional.mse_loss(out["pred"], y)
    return coral_loss(out["pred"], y)


def evaluate(model: CMFModel, samples, device: str) -> dict:
    """自然存在模态（无人工 dropout）+ eval 均值图像 → 指标与预测。"""
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for i in range(0, len(samples), 64):
            chunk = samples[i:i + 64]
            x, y, missing = collate(chunk, train=False)
            x = {k: v.to(device) for k, v in x.items()}
            pred = model.predict_continuous(x, missing.to(device))
            preds.append(pred.cpu().numpy())
            ys.append(y.numpy())
    y_cont = np.concatenate(preds)
    y_true = np.concatenate(ys).astype(int)
    met = ordinal_metrics(y_true, y_cont)
    met["preds"] = y_cont.tolist()
    met["y_true"] = y_true.tolist()
    return met


def train_fold(
    model: CMFModel, data: FoldData, cfg: TrainConfig, device: str,
    rng: np.random.Generator,
) -> tuple[dict[str, float], list[float]]:
    """阶段C：全组合 modality dropout 任务训练 + 验证集早停。返回 (最优test指标, 历史)。"""
    model.to(device)
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.train_lr, weight_decay=cfg.weight_decay,
    )
    combos = [c.to(device) for c in ALL_COMBOS]
    history: list[float] = []
    best = {"qwk": -2.0, "mae": 99.0, "state": None}
    patience, bad = 30, 0

    for ep in range(cfg.train_epochs):
        model.train()
        opt.zero_grad()
        x, y, missing = collate(data.train, train=True, rng=rng)
        x = {k: v.to(device) for k, v in x.items()}
        y, missing = y.to(device), missing.to(device)
        losses = []
        for keep in combos:                       # 15 种组合全部显式监督
            miss = missing | ~keep                # 人工drop ∪ 自然缺失
            losses.append(_task_loss(model, x, y, miss, model.head))
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()
        history.append(float(loss.item()))

        val = evaluate(model, data.val, device)
        if (val["qwk"], -val["mae"]) > (best["qwk"], -best["mae"]):
            best = {"qwk": val["qwk"], "mae": val["mae"],
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"    [train] early stop at ep{ep+1} (best val QWK={best['qwk']:.3f})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return evaluate(model, data.test, device), history
