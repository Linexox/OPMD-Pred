# -*- coding: utf-8 -*-
"""序数分级指标：QWK / MAE / 1-off accuracy / macro-F1。"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score


def round_preds(y_cont: np.ndarray, n_classes: int = 4) -> np.ndarray:
    return np.clip(np.rint(y_cont), 0, n_classes - 1).astype(int)


def ordinal_metrics(y_true: np.ndarray, y_cont: np.ndarray) -> dict[str, float]:
    """y_true: (N,) int 0..3；y_cont: (N,) 连续预测（MSE 输出或 CORAL 期望值）。"""
    yp = round_preds(y_cont)
    return {
        "qwk": float(cohen_kappa_score(y_true, yp, labels=[0, 1, 2, 3], weights="quadratic")),
        "mae": float(np.mean(np.abs(yp - y_true))),
        "mae_cont": float(np.mean(np.abs(y_cont - y_true))),
        "acc_1off": float(np.mean(np.abs(yp - y_true) <= 1)),
        "macro_f1": float(f1_score(y_true, yp, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        # 坍塌诊断：预测几乎不动（std→0）或整体偏移过大时，前 5 项指标会"看起来还行但毫无判别力"
        "pred_std": float(np.std(y_cont)),
        "pred_mean": float(np.mean(y_cont)),
    }


def mean_std_report(per_fold: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = per_fold[0].keys()
    return {
        k: {"mean": float(np.mean([r[k] for r in per_fold])), "std": float(np.std([r[k] for r in per_fold]))}
        for k in keys
    }
