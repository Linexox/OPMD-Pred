# -*- coding: utf-8 -*-
"""阶段B + 阶段C：患者级 5 折 CV，对比预训练 → 全组合同步 dropout 任务训练。

用法（远程 GPU）：
    uv run python scripts/train_cv.py --tag exp1_mse --head mse --device cuda
    uv run python scripts/train_cv.py --tag exp2_coral --head coral
    uv run python scripts/train_cv.py --tag exp3_nopre --head mse --skip-pretrain

产出（artifacts/<tag>/）：
    per_fold.json            每折 test 指标 + val 指标
    summary.json             mean±std
    preds_fold{k}.csv        每折测试集预测（画图/混淆矩阵用）
    models/fold{k}.pt        该折最优权重（早停选出的 val QWK 最优）
    scalers/fold{k}.joblib   该折训练折拟合的 scaler（推理复现用）
    training_history.json    逐 epoch 任务损失

同时把 mean±std 追加到 reports/RESULTS.md（便于横向对比消融实验）。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from opmd_pred.config import CVConfig, ModelConfig, Paths, TrainConfig
from opmd_pred.data.records import load_labeled_records
from opmd_pred.data.split import make_folds, summarize_folds
from opmd_pred.models.cmf_model import CMFModel
from opmd_pred.training.dataset import build_fold_data, fit_preprocessor
from opmd_pred.training.metrics import mean_std_report
from opmd_pred.training.trainer import evaluate, pretrain_fold, train_fold

# 坍塌诊断：pred_std 接近 0 = 模型退化成常数预测，此时 QWK/MAE 会"看起来还行"但毫无判别力
METRIC_KEYS = ["qwk", "mae", "mae_cont", "acc_1off", "macro_f1", "pred_std"]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OPMD-Pred 5 折 CV 训练（阶段B + 阶段C）")
    p.add_argument("--tag", required=True, help="实验名，产出目录 artifacts/<tag>")
    p.add_argument("--head", choices=["mse", "coral"], default="mse")
    p.add_argument("--device", default=None, help="cuda / cpu，默认自动选择")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrain-epochs", type=int, default=None)
    p.add_argument("--train-epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="覆盖阶段C学习率")
    p.add_argument("--patience", type=int, default=None, help="早停耐心（单位=步，全批量下 1 epoch=1 步）")
    p.add_argument("--folds", default=None, help="只跑指定折，逗号分隔，如 0,1,2")
    p.add_argument("--cache-dir", default=None, help="图像嵌入缓存目录，默认 features/image_cache")
    p.add_argument("--skip-pretrain", action="store_true", help="消融：跳过阶段B")
    p.add_argument("--no-results-md", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    paths = Paths()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    cfg_train = TrainConfig(head=args.head)
    if args.pretrain_epochs is not None:
        cfg_train.pretrain_epochs = args.pretrain_epochs
    if args.train_epochs is not None:
        cfg_train.train_epochs = args.train_epochs
    if args.lr is not None:
        cfg_train.train_lr = args.lr
    if args.patience is not None:
        cfg_train.early_stop_patience = args.patience

    out_dir = paths.artifacts / args.tag
    for sub in ("models", "scalers"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else paths.features / "image_cache"
    has_cache = (cache_dir / "image_embeddings.npz").exists()
    print(f"[setup] device={device} head={args.head} pretrain={not args.skip_pretrain}")
    print(f"[setup] image cache: {cache_dir} ({'found' if has_cache else 'MISSING -> 图像模态全缺失'})")

    records = load_labeled_records(paths.data)
    print(f"[records] 标注记录 {len(records)} 条；有 CSV {sum(r.csv_row is not None for r in records)} 条；"
          f"有病灶图 {sum(bool(r.lesion_images) for r in records)} 条")

    cv_cfg = CVConfig(n_splits=args.n_splits, val_ratio=args.val_ratio, seed=args.seed)
    folds = make_folds(records, cv_cfg)
    wanted = None
    if args.folds:
        wanted = {int(x) for x in args.folds.split(",") if x.strip()}

    per_fold: list[dict] = []
    for fold_summary, fold in zip(summarize_folds(folds, records), folds):
        if wanted is not None and fold.fold not in wanted:
            continue
        print(f"\n===== fold {fold.fold} (train {len(fold.train)} / val {len(fold.val)} "
              f"/ test {len(fold.test)}) =====")

        torch.manual_seed(args.seed + fold.fold)
        rng = np.random.default_rng(args.seed + fold.fold)

        prep = fit_preprocessor(records, fold)
        prep.save(out_dir / "scalers" / f"fold{fold.fold}.joblib")
        data = build_fold_data(records, fold, prep, cache_dir if has_cache else None)

        model = CMFModel(ModelConfig(), head=args.head)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [model] 可训练参数 {n_params:,}")

        if not args.skip_pretrain:
            pretrain_fold(model, data, cfg_train, device, rng)
        metric, history = train_fold(model, data, cfg_train, device, rng)

        torch.save(model.state_dict(), out_dir / "models" / f"fold{fold.fold}.pt")
        (out_dir / f"preds_fold{fold.fold}.csv").write_text(
            "patient_key,y_true,y_pred_cont,y_pred_round\n"
            + "\n".join(
                f"{s.patient_key},{int(m)},{c:.4f},{int(round(c))}"
                for s, m, c in zip(data.test, metric["y_true"], metric["preds"])
            ),
            encoding="utf-8",
        )
        (out_dir / f"history_fold{fold.fold}.json").write_text(
            json.dumps(history), encoding="utf-8")

        row = {"fold": fold.fold, **{k: float(metric[k]) for k in METRIC_KEYS}}
        row["pred_mean"] = float(metric["pred_mean"])
        val = evaluate(model, data.val, device)
        row.update({f"val_{k}": float(val[k]) for k in METRIC_KEYS})
        per_fold.append(row)
        print("  [test] " + "  ".join(f"{k}={row[k]:.4f}" for k in METRIC_KEYS))
        if row["pred_std"] < 0.1:
            print(f"  [WARN] 本折预测近乎常数（std={row['pred_std']:.4f}, "
                  f"mean={row['pred_mean']:.4f}）→ 模型未学到判别信息，指标不可信")

    if not per_fold:
        print("[warn] 没有跑任何折")
        return

    summary = mean_std_report([{k: v for k, v in r.items() if k != "fold"} for r in per_fold])
    (out_dir / "per_fold.json").write_text(json.dumps(per_fold, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n===== {args.tag} ({args.head}, pretrain={not args.skip_pretrain}) =====")
    print(f"{'metric':<10}{'mean':>10}{'std':>10}")
    for k in METRIC_KEYS:
        print(f"{k:<10}{summary[k]['mean']:>10.4f}{summary[k]['std']:>10.4f}")
    print(f"\n[out] {out_dir}")

    if not args.no_results_md:
        md = paths.reports / "RESULTS.md"
        md.parent.mkdir(parents=True, exist_ok=True)
        head_exist = md.exists()
        with md.open("a", encoding="utf-8") as f:
            if not head_exist:
                f.write("# 实验结果汇总\n\n")
                f.write(f"{'run':<24}{'head':<8}{'pretrain':<10}"
                        + "".join(f"{k:>12}" for k in METRIC_KEYS) + "\n")
            f.write(f"{args.tag:<24}{args.head:<8}{str(not args.skip_pretrain):<10}"
                    + "".join(f"{summary[k]['mean']:.4f}±{summary[k]['std']:.3f}".rjust(12)
                              for k in METRIC_KEYS) + "\n")
        print(f"[out] 已追加 {md} （{datetime.now():%Y-%m-%d %H:%M}）")


if __name__ == "__main__":
    main()
