# -*- coding: utf-8 -*-
"""阶段D：可解释性 —— InterSHAP（模态级跨模态交互） + 表格特征级精确 Shapley。

用法（远程）：
    uv run python scripts/explain.py --tag exp1_mse --device cuda
    uv run python scripts/explain.py --tag exp1_mse --folds 0 --max-samples 20

产出（reports/explain/<tag>/）：
    intershap_global.json        global InterSHAP 分数 + 4×4 平均|Φ|矩阵
    intershap_local.csv          每个测试样本的 local InterSHAP
    intershap_matrix.png         4×4 |Φ| 热图
    modality_contribution.png    对角（模态自身）与非对角（交互）贡献条形图
    shap_feature_importance.csv  12 个表格特征 |φ| 均值排名（含中文原名）
    shap_feature_importance.png
    shap_per_sample.csv          逐样本 12 维 Shapley 值

关键设计：屏蔽模态 = 替换为模型训练时用的 learnable missing embedding，
因此不像 InterSHAP 原文那样需要"拿别的样本替换"，不存在 masking 分布外扰动。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from opmd_pred.config import CVConfig, ModelConfig, Paths
from opmd_pred.data.preprocessing import TabularPreprocessor
from opmd_pred.data.records import load_labeled_records
from opmd_pred.data.split import make_folds
from opmd_pred.explain.intershap import global_intershap, intershap
from opmd_pred.explain.shap_analysis import (
    modality_group_shapley,
    subsample_background,
    tabular_shapley,
)
from opmd_pred.models.cmf_model import MODALITY_ORDER, CMFModel
from opmd_pred.training.dataset import TABULAR_FEATURES, build_fold_data

# matplotlib 图用英文名（Linux 环境通常缺中文字体），CSV 保留中文原名
FEATURE_EN = {
    "病原体": "Pathogen",
    "炎细胞": "Inflammation",
    "TCT意见": "TCT opinion",
    "分析上皮细胞数": "Analyzed cells",
    "二倍体细胞数量": "Diploid count",
    "四倍体细胞数量": "Tetraploid count",
    "异倍体细胞数量": "Aneuploid count",
    "高倍体细胞数量": "High-ploidy count",
    "非整倍体细胞数量": "Non-diploid count",
    "DI>=2.3细胞": "DI>=2.3 count",
    "甲基化总数": "Methylation",
    "age": "Age",
}
MODALITY_EN = {
    "image": "Image",
    "exfoliative": "Exfoliative",
    "methylation": "Methylation",
    "basic": "Demographics",
}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OPMD-Pred 可解释性分析（阶段D）")
    p.add_argument("--tag", required=True, help="train_cv.py 里用的实验名")
    p.add_argument("--head", choices=["mse", "coral"], default="mse",
                   help="必须与 train_cv.py 训练该 tag 时用的 head 一致")
    p.add_argument("--device", default=None)
    p.add_argument("--folds", default=None, help="只跑指定折，逗号分隔")
    p.add_argument("--n-bg", type=int, default=24, help="SHAP 背景样本数（从训练折抽）")
    p.add_argument("--max-samples", type=int, default=None, help="每折最多解释多少个测试样本")
    p.add_argument("--shap-samples", type=int, default=None, help="每折最多算多少个样本的 SHAP")
    p.add_argument("--artifacts", default=None)
    p.add_argument("--cache-dir", default=None, help="图像嵌入缓存目录，默认 features/image_cache")
    return p


def _save_intershap_figs(matrix: np.ndarray, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [MODALITY_EN[m] for m in MODALITY_ORDER]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(matrix, cmap="Reds")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if matrix[i, j] < matrix.max() * 0.6 else "white")
    ax.set_title(r"Mean $|\Phi_{ij}|$ across test samples")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "intershap_matrix.png", dpi=180)
    plt.close(fig)

    off = matrix.sum() - np.diag(matrix).sum()
    per_mod_interaction = np.array([
        sum(matrix[i, j] for j in range(len(labels)) if j != i) for i in range(len(labels))
    ])
    own = np.diag(matrix)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(x - 0.2, own, 0.4, label=r"own contribution $\Phi_{ii}$", color="#4878a8")
    ax.bar(x + 0.2, per_mod_interaction, 0.4, label="interaction roles " + r"$\sum_{j\neq i}|\Phi_{ij}|$",
           color="#c44e52")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.legend(fontsize=8)
    ax.set_ylabel(r"mean $|\Phi|$")
    fig.tight_layout()
    fig.savefig(out_dir / "modality_contribution.png", dpi=180)
    plt.close(fig)


def _save_shap_figs(mean_abs: np.ndarray, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names_en = [FEATURE_EN.get(f, f) for f in TABULAR_FEATURES]
    order = np.argsort(mean_abs)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh([names_en[i] for i in order], mean_abs[order], color="#4878a8")
    ax.set_xlabel(r"mean $|\phi|$ (Shapley value)")
    ax.set_title("Tabular feature importance")
    fig.tight_layout()
    fig.savefig(out_dir / "shap_feature_importance.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_argparser().parse_args()
    paths = Paths()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    art_dir = Path(args.artifacts) if args.artifacts else paths.artifacts / args.tag
    out_dir = paths.reports / "explain" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_labeled_records(paths.data)
    cv_cfg = CVConfig()
    folds = make_folds(records, cv_cfg)
    wanted = {int(x) for x in args.folds.split(",") if x.strip()} if args.folds else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else paths.features / "image_cache"
    has_cache = (cache_dir / "image_embeddings.npz").exists()

    all_local: list[float] = []
    all_matrix: list[np.ndarray] = []
    all_shap: list[np.ndarray] = []
    complete_scores: list[np.ndarray] = []

    for fold in folds:
        if wanted is not None and fold.fold not in wanted:
            continue
        ckpt = art_dir / "models" / f"fold{fold.fold}.pt"
        if not ckpt.exists():
            print(f"[warn] 缺 {ckpt}，跳过 fold {fold.fold}")
            continue
        scaler_path = art_dir / "scalers" / f"fold{fold.fold}.joblib"
        prep = TabularPreprocessor.load(scaler_path) if scaler_path.exists() \
            else None
        if prep is None:
            from opmd_pred.training.dataset import fit_preprocessor
            prep = fit_preprocessor(records, fold)
        data = build_fold_data(records, fold, prep, cache_dir if has_cache else None)

        model = CMFModel(ModelConfig(), head=args.head)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device).eval()
        print(f"\n===== explain fold {fold.fold}: train {len(data.train)} "
              f"/ val {len(data.val)} / test {len(data.test)} =====")

        # ---- InterSHAP（全部测试样本，每样本 16 次前向） ----
        test_samples = data.test
        if args.max_samples:
            test_samples = test_samples[: args.max_samples]
        res = intershap(model, test_samples, device, verbose=True)
        g_all, matrix = global_intershap(res, complete_case_only=False)
        g_cc, matrix_cc = global_intershap(res, complete_case_only=True)
        print(f"  [intershap] local mean={res.intershap_local.mean():.4f} "
              f"± {res.intershap_local.std():.4f} | global={g_all:.4f} "
              f"| complete-case={g_cc:.4f} (n={int(res.complete_case_mask.sum())})")
        all_local.append(res.intershap_local)
        all_matrix.append(matrix)

        with (out_dir / f"intershap_local_fold{fold.fold}.csv").open(
                "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_key", "intershap_local", "complete_case"]
                       + [f"own_{m}" for m in MODALITY_ORDER])
            for s, loc, cc, phi in zip(test_samples, res.intershap_local,
                                       res.complete_case_mask, res.phi_matrix):
                w.writerow([s.patient_key, f"{loc:.6f}", int(cc)]
                           + [f"{v:.6f}" for v in np.diag(phi)])
                if cc:
                    complete_scores.append(phi)

        # ---- 表格特征级精确 Shapley ----
        bg = subsample_background(data.train, max_n=args.n_bg, seed=fold.fold)
        shap_samples = test_samples[: args.shap_samples] if args.shap_samples else test_samples
        rows, group_rows = [], []
        for s in shap_samples:
            phi = tabular_shapley(model, s, bg, device)
            rows.append(phi)
            grp = modality_group_shapley(phi)
            group_rows.append(grp)
            print(f"    [shap] {s.patient_key} done")
        phis = np.stack(rows) if rows else np.zeros((0, len(TABULAR_FEATURES)))
        if len(phis):
            all_shap.append(phis)
            with (out_dir / f"shap_per_sample_fold{fold.fold}.csv").open(
                    "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["patient_key", "y_true"] + list(TABULAR_FEATURES))
                for s, phi in zip(shap_samples, phis):
                    w.writerow([s.patient_key, int(s.y)] + [f"{v:.6f}" for v in phi])

    # ---- 汇总 ----
    if not all_matrix:
        print("[error] 没有任何折成功，检查 --tag 是否正确")
        return

    mean_matrix = np.stack(all_matrix).mean(axis=0)
    pooled_local = np.concatenate(all_local)
    summary = {
        "tag": args.tag,
        "n_samples": int(len(pooled_local)),
        "intershap_local_mean": float(pooled_local.mean()),
        "intershap_local_std": float(pooled_local.std()),
    }
    # 对 mean|Φ| 矩阵再算一次比值，得到全局分数
    total = mean_matrix.sum()
    off = total - np.diag(mean_matrix).sum()
    summary["intershap_global"] = float(off / total) if total > 1e-12 else 0.0
    summary["mean_abs_phi_matrix"] = mean_matrix.tolist()
    summary["modality_order"] = MODALITY_ORDER
    if complete_scores:
        cc_mean = np.abs(np.stack(complete_scores)).mean(axis=0)
        summary["intershap_global_complete_case"] = float(
            (cc_mean.sum() - np.diag(cc_mean).sum()) / max(cc_mean.sum(), 1e-12))
        summary["n_complete_case"] = int(len(complete_scores))

    (out_dir / "intershap_global.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    _save_intershap_figs(mean_matrix, out_dir)

    if all_shap:
        all_phi = np.concatenate(all_shap, axis=0)
        mean_abs = np.abs(all_phi).mean(axis=0)
        order = np.argsort(-mean_abs)
        with (out_dir / "shap_feature_importance.csv").open(
                "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "feature", "feature_en", "modality", "mean_abs_phi", "mean_phi"])
            for r, i in enumerate(order, 1):
                mod = ("exfoliative" if i < 10 else "methylation" if i == 10 else "basic")
                w.writerow([r, TABULAR_FEATURES[i], FEATURE_EN.get(TABULAR_FEATURES[i], ""),
                            mod, f"{mean_abs[i]:.6f}", f"{all_phi[:, i].mean():.6f}"])
        _save_shap_figs(mean_abs, out_dir)
        summary["top_features"] = [
            {"feature": TABULAR_FEATURES[i], "mean_abs_phi": float(mean_abs[i])}
            for i in order[:5]
        ]
        (out_dir / "intershap_global.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[summary] InterSHAP global = {summary['intershap_global']:.4f} "
          f"(local mean {summary['intershap_local_mean']:.4f} ± {summary['intershap_local_std']:.4f})")
    print(f"[out] {out_dir}")


if __name__ == "__main__":
    main()
