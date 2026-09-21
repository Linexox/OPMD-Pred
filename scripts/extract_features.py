# -*- coding: utf-8 -*-
"""阶段A：全量图像嵌入提取（冻结 ViT，1 eval + n_aug 增强 / 张）。

用法: python scripts/extract_features.py [--device cuda|cpu] [--n-aug 10]
产出: features/image_cache/image_embeddings.npz + image_index.json
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from opmd_pred.config import ImageConfig, Paths
from opmd_pred.data.records import load_labeled_records
from opmd_pred.data.features import extract_image_embeddings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-aug", type=int, default=None, help="覆盖默认增强倍数")
    args = ap.parse_args()

    paths = Paths()
    records = load_labeled_records(paths.data)
    n_with = sum(1 for r in records if r.lesion_images)
    n_img = sum(len(r.lesion_images) for r in records)
    print(f"[records] 标注记录 {len(records)} 条, 有病灶图 {n_with} 条, 共 {n_img} 张")

    cfg = ImageConfig()
    if args.n_aug is not None:
        cfg.n_aug = args.n_aug
    out_dir = paths.features / "image_cache"
    print(f"[extract] device={args.device} model={cfg.model_id} n_aug={cfg.n_aug} -> {out_dir}")
    npz = extract_image_embeddings(records, out_dir, cfg, device=args.device)
    print(f"[done] {npz}")


if __name__ == "__main__":
    main()
