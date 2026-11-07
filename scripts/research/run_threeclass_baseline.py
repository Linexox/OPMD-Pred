"""Run a three-class baseline using the current mainline multimodal inputs."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from run_methylation_cell_transfer import Fusion, Inputs, seed
from sysu_binary_common import sha, write


LABELS = ["正常/轻度", "中度", "重度"]


class ThreeClassFusion(Fusion):
    """The current Fusion model with a three-class output head."""

    def __init__(self, table_size, config):
        super().__init__(table_size, config)
        self.head = nn.Sequential(nn.LayerNorm(config["dimension"]), nn.Linear(config["dimension"], 3))


class ThreeClassInputs(Inputs):
    def pack(self, ids, device, train=False):
        values = super().pack(ids, device, train)
        ix = np.array([self.index[s] for s in ids])
        values["target"] = torch.as_tensor(self.grade[ix], dtype=torch.long, device=device)
        return values


def balanced_weights(labels, device):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3)
    if (counts == 0).any():
        raise ValueError(f"Training fold has an empty class: {counts.tolist()}")
    return torch.tensor(len(labels) / (3 * counts), dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model, inputs, ids, device, fold):
    model.eval()
    rows = []
    for sid in ids:
        i = inputs.index[sid]
        probabilities = None
        if inputs.present[i].any():
            probabilities = model(inputs.pack([sid], device))[0].softmax(0).cpu().tolist()
        rows.append({
            "sample_id": sid,
            "grade": int(inputs.grade[i]),
            "fold": fold,
            "probabilities": probabilities,
            "prediction": int(np.argmax(probabilities)) if probabilities is not None else None,
            "cell_present": bool(inputs.present[i, 3]),
            "common_evaluable": bool(inputs.common[i]),
        })
    return rows


def _f1(recall, precision):
    return 2 * recall * precision / (recall + precision) if recall + precision else 0.0


def score_three(rows):
    valid = [r for r in rows if r["probabilities"] is not None]
    abstain = [sum(r["probabilities"] is None and r["grade"] == k for r in rows) for k in range(3)]
    y = np.asarray([r["grade"] for r in valid], dtype=np.int64)
    pred = np.asarray([r["prediction"] for r in valid], dtype=np.int64)
    cm = np.zeros((3, 3), dtype=int)
    for actual, predicted in zip(y, pred):
        cm[actual, predicted] += 1
    recalls = []
    f1s = []
    for k in range(3):
        tp = int(cm[k, k])
        support = int(cm[k].sum())
        predicted_count = int(cm[:, k].sum())
        recall = tp / support if support else None
        precision = tp / predicted_count if predicted_count else 0.0
        recalls.append(recall)
        f1s.append(_f1(recall or 0.0, precision) if recall is not None else None)
    return {
        "assigned_count": len(rows),
        "eligible_count": len(valid),
        "coverage": len(valid) / len(rows) if rows else 0.0,
        "abstain_by_grade": abstain,
        "accuracy": float((pred == y).mean()) if len(valid) else None,
        "recalls": recalls,
        "macro_recall": float(np.mean([x for x in recalls if x is not None])) if valid else None,
        "f1_by_grade": f1s,
        "macro_f1": float(np.mean([x for x in f1s if x is not None])) if valid else None,
        "mae": float(np.abs(pred - y).mean()) if valid else None,
        "confusion": cm.tolist(),
        "original_grade_destinations": cm.tolist(),
    }


def score_binary_merge(rows):
    valid = [r for r in rows if r["probabilities"] is not None]
    actual = np.asarray([int(r["grade"] > 0) for r in valid], dtype=np.int64)
    pred = np.asarray([int(sum(r["probabilities"][1:]) >= 0.5) for r in valid], dtype=np.int64)
    cm = np.zeros((2, 2), dtype=int)
    for a, p in zip(actual, pred):
        cm[a, p] += 1
    recalls = [float(cm[k, k] / cm[k].sum()) if cm[k].sum() else None for k in range(2)]
    return {
        "accuracy": float((pred == actual).mean()) if len(valid) else None,
        "normal_mild_recall": recalls[0],
        "moderate_or_above_recall": recalls[1],
        "confusion": cm.tolist(),
        "original_grade_destinations": [
            [int(((np.asarray([r["grade"] for r in valid]) == k) & (pred == j)).sum()) for j in range(2)]
            for k in range(3)
        ],
    }


def fit(inputs, train_ids, val_ids, methyl_encoder, config, initial_seed, epochs, device, folder):
    seed(initial_seed)
    model = ThreeClassFusion(inputs.table.shape[1], config).to(device)
    if methyl_encoder is not None:
        model.methyl.load_state_dict(methyl_encoder, strict=True)
    methyl_ids = {id(p) for p in model.methyl.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in methyl_ids], "lr": config["lr"]},
        {"params": model.methyl.parameters(), "lr": config["methylation_lr"]},
    ], weight_decay=config["weight_decay"])
    ids = inputs.eligible(train_ids)
    labels = [inputs.grade[inputs.index[s]] for s in ids]
    class_weight = balanced_weights(labels, device)
    history = []
    best = (-1.0, -float("inf"))
    selected = 1
    stale = 0
    for epoch in range(1, epochs + 1):
        model.methyl.requires_grad_(epoch > config["warmup_epochs"])
        model.train()
        order = torch.randperm(len(ids)).tolist()
        total = 0.0
        for start in range(0, len(ids), config["batch_size"]):
            chunk = [ids[i] for i in order[start:start + config["batch_size"]]]
            batch = inputs.pack(chunk, device, True)
            loss = F.cross_entropy(model(batch), batch["target"], weight=class_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient norm")
            optimizer.step()
            total += float(loss.detach()) * len(chunk)
        entry = {"epoch": epoch, "train_loss": total / len(ids)}
        if val_ids is not None:
            rows = predict(model, inputs, val_ids, device, -1)
            metrics = score_three(rows)
            valid = [r for r in rows if r["probabilities"] is not None]
            probabilities = torch.tensor([r["probabilities"] for r in valid], device=device)
            targets = torch.tensor([r["grade"] for r in valid], dtype=torch.long, device=device)
            validation_loss = float(F.nll_loss(probabilities.clamp_min(1e-8).log(), targets, weight=class_weight))
            key = (metrics["macro_recall"], -validation_loss)
            entry.update(validation=metrics, validation_loss=validation_loss)
            if epoch > config["warmup_epochs"]:
                if key > best:
                    best, selected, stale = key, epoch, 0
                else:
                    stale += 1
        else:
            selected = epoch
        history.append(entry)
        if val_ids is not None and stale >= config["fusion_patience"] and epoch > config["warmup_epochs"]:
            break
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "training.json", {
        "history": history,
        "selected_epoch": selected,
        "training_ids": ids,
        "validation_ids": val_ids,
        "normalization_mean": inputs.mean.tolist(),
        "normalization_scale": inputs.scale.tolist(),
        "tct_vocab": inputs.vocab,
        "task": "three_class",
    })
    if val_ids is None:
        torch.save({
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": config,
            "table_size": inputs.table.shape[1],
            "normalization_mean": inputs.mean.tolist(),
            "normalization_scale": inputs.scale.tolist(),
            "tct_vocab": inputs.vocab,
            "training_ids": ids,
            "task": "three_class",
        }, folder / "classifier.pt")
    return model, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--methyl-checkpoint", type=Path, required=True)
    parser.add_argument("--cell-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    bundle = json.loads((args.data / "bundle.json").read_text(encoding="utf-8"))
    photos = np.load(args.data / "photos.npz", allow_pickle=False)
    cell_cache = np.load(args.cell_features, allow_pickle=False)
    if "sample_ids" in cell_cache:
        cell_order = list(cell_cache["sample_ids"])
        expected = [r["sample_id"] for r in bundle["samples"]]
        if cell_order != expected:
            lookup = {sid: i for i, sid in enumerate(cell_order)}
            cells = cell_cache["values"][[lookup[s] for s in expected]].astype(np.float32)
        else:
            cells = cell_cache["values"].astype(np.float32)
    else:
        cells = cell_cache["values"].astype(np.float32)
    if cells.shape != (len(bundle["samples"]), 512):
        raise ValueError(f"Unexpected cell feature shape: {cells.shape}")
    methyl = torch.load(args.methyl_checkpoint, map_location="cpu", weights_only=True)
    if methyl["genes"] != bundle["genes"]:
        raise ValueError("Methylation gene order differs from the prepared bundle")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.time()
    write(args.output / "configuration.json", {
        "config": config,
        "bundle_sha256": sha(args.data / "bundle.json"),
        "photo_sha256": sha(args.data / "photos.npz"),
        "cell_features_sha256": sha(args.cell_features),
        "methylation_checkpoint_sha256": sha(args.methyl_checkpoint),
        "sources": {
            "run_threeclass_baseline.py": sha(Path(__file__)),
            "run_methylation_cell_transfer.py": sha(Path(__file__).with_name("run_methylation_cell_transfer.py")),
            "sysu_binary_common.py": sha(Path(__file__).with_name("sysu_binary_common.py")),
        },
        "task": "three_class",
        "labels": LABELS,
    })
    reports = {"folds": [], "rows": []}
    for fold in bundle["folds"]:
        number = fold["fold"]
        inner_train = fold["inner_train"]["sample_ids"]
        inner_val = fold["inner_val"]["sample_ids"]
        outer_train = fold["outer_train"]["sample_ids"]
        holdout = fold["outer_holdout"]["sample_ids"]
        folder = args.output / f"fold{number}"
        inputs = ThreeClassInputs(bundle, photos, cells, inner_train, True)
        _, selected = fit(inputs, inner_train, inner_val, methyl["encoder"], config,
                          config["seed"] + number, config["fusion_epochs"], args.device, folder / "inner")
        inputs = ThreeClassInputs(bundle, photos, cells, outer_train, True)
        model, _ = fit(inputs, outer_train, None, methyl["encoder"], config,
                       config["seed"] + 100 + number, selected, args.device, folder / "refit")
        rows = predict(model, inputs, holdout, args.device, number)
        saved = torch.load(folder / "refit" / "classifier.pt", map_location="cpu", weights_only=True)
        restored = ThreeClassFusion(inputs.table.shape[1], config).to(args.device)
        restored.load_state_dict(saved["state"], strict=True)
        reloaded = predict(restored, inputs, holdout, args.device, number)
        reload_error = max([
            abs(a - b)
            for row, other in zip(rows, reloaded)
            if row["probabilities"] is not None
            for a, b in zip(row["probabilities"], other["probabilities"])
        ] or [0.0])
        if reload_error >= 1e-6:
            raise AssertionError(f"Reload prediction mismatch in fold {number}: {reload_error}")
        metrics = score_three(rows)
        binary = score_binary_merge(rows)
        item = {"fold": number, "selected_epoch": selected, "metrics": metrics,
                "binary_merge": binary, "reload_error": reload_error}
        reports["folds"].append(item)
        reports["rows"].extend(rows)
        write(folder / "predictions.json", {"rows": rows, **item})
        print(f"three_class fold{number} epoch={selected} acc={metrics['accuracy']:.4f} macro_recall={metrics['macro_recall']:.4f}", flush=True)
    if len(reports["rows"]) != 117:
        raise AssertionError("The five folds did not cover all 117 development records")
    reports["metrics"] = score_three(reports["rows"])
    reports["binary_merge"] = score_binary_merge(reports["rows"])
    reports["common_input_metrics"] = score_three([r for r in reports["rows"] if r["common_evaluable"]])
    reports["fold_mean_accuracy"] = float(np.mean([f["metrics"]["accuracy"] for f in reports["folds"]]))
    reports["fold_max_accuracy"] = float(max(f["metrics"]["accuracy"] for f in reports["folds"]))
    reports["elapsed_seconds"] = time.time() - started
    reports["task"] = "three_class"
    write(args.output / "results.json", reports)
    print("THREE CLASS COMPLETED", flush=True)


if __name__ == "__main__":
    main()
