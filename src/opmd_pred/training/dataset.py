# -*- coding: utf-8 -*-
"""折数据组装：records + fold + (训练折拟合的)preprocessor + 图像缓存 → 训练样本。

表格 12 维拼接顺序（SHAP/报告统一用这个顺序）：
  EXFOLIATIVE_FEATURES(10) + METHYLATION_FEATURES(1) + BASIC_FEATURES(1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import (
    BASIC_FEATURES,
    EXFOLIATIVE_FEATURES,
    METHYLATION_FEATURES,
    MODALITIES,
)
from ..data.features import load_image_cache
from ..data.preprocessing import TabularPreprocessor
from ..data.records import Record
from ..data.split import Fold
from ..models.cmf_model import MODALITY_ORDER, MODALITY_IDX

TABULAR_FEATURES = EXFOLIATIVE_FEATURES + METHYLATION_FEATURES + BASIC_FEATURES
_TAB_SLICES = {
    "exfoliative": slice(0, len(EXFOLIATIVE_FEATURES)),
    "methylation": slice(len(EXFOLIATIVE_FEATURES), len(EXFOLIATIVE_FEATURES) + len(METHYLATION_FEATURES)),
    "basic": slice(len(EXFOLIATIVE_FEATURES) + len(METHYLATION_FEATURES), len(TABULAR_FEATURES)),
}


@dataclass
class Sample:
    patient_key: str
    name: str
    y: float
    tab: np.ndarray                     # (12,) 已 scaling
    present_tab: dict[str, bool]
    has_image: bool
    image_pool: np.ndarray | None = None  # (n, 768) train 期随机采样（含增强）
    image_eval: np.ndarray | None = None  # (768,) 评估期确定性嵌入（eval 均值）

    def natural_missing(self) -> np.ndarray:
        """(4,) bool：True=缺失。顺序 = MODALITY_ORDER。"""
        m = np.zeros(len(MODALITY_ORDER), dtype=bool)
        m[MODALITY_IDX["image"]] = not self.has_image
        for mod, ok in self.present_tab.items():
            m[MODALITY_IDX[mod]] = not ok
        return m


@dataclass
class FoldData:
    fold: int
    train: list[Sample] = field(default_factory=list)
    val: list[Sample] = field(default_factory=list)
    test: list[Sample] = field(default_factory=list)


def _build_sample(r: Record, prep: TabularPreprocessor, cache: dict, cache_meta: dict) -> Sample:
    if r.csv_row is not None:
        vecs, present = prep.transform(r.csv_row)
        tab = np.concatenate(
            [vecs["exfoliative"], vecs["methylation"], vecs["basic"]]
        ).astype(np.float32)
    else:
        tab = np.zeros(len(TABULAR_FEATURES), dtype=np.float32)
        present = {"exfoliative": False, "methylation": False, "basic": False}

    pool = eval_mean = None
    entry = cache_meta.get("patients", {}).get(r.patient_key)
    if entry and entry.get("keys"):
        embs = np.stack([cache[k] for k in entry["keys"]])            # (n_img*(1+n_aug), 768)
        eval_keys = [k for k in entry["keys"] if k.endswith("|eval")]
        pool = embs.astype(np.float32)
        if eval_keys:
            eval_mean = np.mean([cache[k] for k in eval_keys], axis=0).astype(np.float32)
        else:
            eval_mean = embs.mean(axis=0).astype(np.float32)

    return Sample(
        patient_key=r.patient_key,
        name=r.name,
        y=float(r.label_idx),
        tab=tab,
        present_tab=present,
        has_image=pool is not None,
        image_pool=pool,
        image_eval=eval_mean,
    )


def build_fold_data(
    records: list[Record],
    fold: Fold,
    prep: TabularPreprocessor,
    cache_dir: Path | str | None = None,
) -> FoldData:
    cache, meta = ({}, {"patients": {}})
    if cache_dir is not None and Path(cache_dir, "image_embeddings.npz").exists():
        cache, meta = load_image_cache(cache_dir)
    by_key = {r.patient_key: r for r in records}
    fd = FoldData(fold=fold.fold)
    for part in ("train", "val", "test"):
        for k in getattr(fold, part):
            fd.__dict__[part].append(_build_sample(by_key[k], prep, cache, meta))
    return fd


def fit_preprocessor(records: list[Record], fold: Fold) -> TabularPreprocessor:
    """只在 fold.train 上拟合（防泄漏）。"""
    by_key = {r.patient_key: r for r in records}
    rows = [by_key[k].csv_row for k in fold.train if by_key[k].csv_row]
    return TabularPreprocessor().fit(rows)


def collate(samples: list[Sample], train: bool, rng: np.random.Generator | None = None):
    """→ (x dict, y tensor, natural_missing (B,4) bool tensor, image (B,768))。

    train=True 时图像从 pool 随机采一张（增强）；否则用确定性 eval 均值。
    """
    import torch

    tab = np.stack([s.tab for s in samples])
    y = np.array([s.y for s in samples], dtype=np.float32)
    missing = np.stack([s.natural_missing() for s in samples])

    imgs = []
    for s in samples:
        if not s.has_image:
            imgs.append(np.zeros(768, dtype=np.float32))
        elif train and rng is not None:
            imgs.append(s.image_pool[rng.integers(len(s.image_pool))])
        else:
            imgs.append(s.image_eval)
    img = np.stack(imgs).astype(np.float32)

    x = {
        "image": img,
        "exfoliative": tab[:, _TAB_SLICES["exfoliative"]],
        "methylation": tab[:, _TAB_SLICES["methylation"]],
        "basic": tab[:, _TAB_SLICES["basic"]],
    }
    return (
        {k: torch.from_numpy(v) for k, v in x.items()},
        torch.from_numpy(y),
        torch.from_numpy(missing),
    )


def samples_to_batches(samples: list[Sample], batch_size: int, train: bool,
                       rng: np.random.Generator | None = None):
    """小批量迭代（当前数据量小，通常整批即可；留接口给未来扩展）。"""
    idx = np.arange(len(samples))
    if train and rng is not None:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [samples[j] for j in idx[i:i + batch_size]]
        yield collate(chunk, train=train, rng=rng)
