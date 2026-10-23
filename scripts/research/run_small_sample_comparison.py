"""Paired image classifiers, geometry augmentation and nested training-size curves."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from opmd_pred.scripts.common import set_seed

SPEC = importlib.util.spec_from_file_location("september_study", Path(__file__).with_name("run_september_comparison.py"))
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


class Classifier(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(3, 1536))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(3)) if kind == "linear" else None
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.))) if kind == "cosine" else None
        self.kind = kind

    def forward(self, features):
        if self.kind == "linear":
            return F.linear(features, self.weight, self.bias)
        return F.normalize(features, dim=1) @ F.normalize(self.weight, dim=1).T * self.log_scale.exp().clamp(max=100)


def subset(ids, fraction, entries, seed=42):
    grouped = {}
    for sid in ids:
        grouped.setdefault(entries[sid]["cv_component"], []).append(sid)
    order = sorted(grouped, key=lambda key: hashlib.sha256(f"training-size:{seed}:{key}".encode()).hexdigest())
    # Nested prefixes; if the requested prefix lacks a class, extend the prefix.
    count = max(1, math.ceil(len(order) * fraction))
    while count <= len(order):
        selected = {sid for group in order[:count] for sid in grouped[group]}
        if {entries[sid]["label"] for sid in selected} == {0, 1, 2}:
            return [sid for sid in ids if sid in selected]
        count += 1
    raise ValueError("Training pool lacks one of the three classes")


def score(model, features, labels, held):
    model.eval()
    with torch.inference_mode():
        probabilities = model(features[held, 0].cuda()).softmax(1).cpu()
    return STUDY.REFERENCE.score(probabilities, labels[held], labels[held].tolist()), probabilities


def train(features, labels, train_indices, full_pool_count, val_indices, kind, augmentation,
          directory, seed, epochs, identity):
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory / "done.json"
    if completed.exists():
        saved = json.loads(completed.read_text())
        STUDY.require(saved["identity"] == identity, "Started image stage changed")
        STUDY.require(STUDY.digest(directory / "classifier.pt") == saved["classifier_sha256"], "Image classifier changed")
        return saved
    STUDY.check_gpu()
    STUDY.check_time()
    set_seed(seed)
    model = Classifier(kind).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = torch.bincount(labels[train_indices], minlength=3)
    STUDY.require(bool((counts > 0).all()), "Training size subset lacks a class")
    weights = (len(train_indices) / (3 * counts.float())).cuda()
    best_key, best_epoch, stale = (-1., -1.), None, 0
    history = []
    for epoch in range(1, epochs + 1):
        STUDY.check_time()
        if val_indices is not None and stale >= 5:
            break
        model.train()
        # Keep the number of updates equal across the training-size fractions.
        if len(train_indices) == full_pool_count:
            indices = train_indices[torch.randperm(len(train_indices))]
        else:
            indices = train_indices[torch.randint(len(train_indices), (full_pool_count,))]
        total_loss = 0.
        for chunk in indices.split(8):
            views = torch.randint(1, 5, (len(chunk),))
            if augmentation == "none":
                views.zero_()
            logits = model(features[chunk, views].cuda())
            loss = F.cross_entropy(logits, labels[chunk].cuda(), weight=weights)
            STUDY.require(torch.isfinite(loss), "Nonfinite image loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(chunk)
        item = dict(epoch=epoch, training_loss=total_loss / len(indices), update_count=math.ceil(len(indices) / 8))
        if val_indices is not None:
            metrics, _ = score(model, features, labels, val_indices)
            item["inner_validation"] = metrics
            m = metrics["fixed_median_0.5"]
            key = m["accuracy_all_assigned"], m["macro_f1"]
            if key > best_key:
                best_key, best_epoch, stale = key, epoch, 0
            else:
                stale += 1
        else:
            best_epoch = epoch
        history.append(item)
        STUDY.REFERENCE.save_torch(directory / "last.pt", dict(state=model.state_dict(), optimizer=optimizer.state_dict(),
                                   epoch=epoch, history=history, identity=identity,
                                   cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
    STUDY.require(best_epoch is not None, "No image training epoch completed")
    STUDY.REFERENCE.save_torch(directory / "classifier.pt", dict(state=model.cpu().state_dict(), identity=identity,
                               kind=kind, class_counts=counts.tolist(), selected_epoch=best_epoch))
    result = dict(identity=identity, class_counts=counts.tolist(), selected_epoch=best_epoch, history=history,
                  classifier_sha256=STUDY.digest(directory / "classifier.pt"),
                  training_records=len(train_indices), per_epoch_updates=math.ceil(full_pool_count / 8))
    STUDY.write(completed, result)
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def variant(args, protocol, kind, augmentation, fraction):
    name = f"{kind}_{augmentation}_fraction{int(fraction * 100)}"
    root = args.output / name
    root.mkdir(parents=True, exist_ok=True)
    with np.load(args.cache / "baseline.npz", allow_pickle=False) as loaded:
        ids = loaded["sample_ids"].tolist()
        present = loaded["image_present"].copy()
        features = loaded["features"].copy()
        labels = torch.from_numpy(loaded["labels"].copy()).long()
    if augmentation == "geometry":
        with np.load(args.features / "geometry.npz", allow_pickle=False) as loaded:
            STUDY.require(loaded["sample_ids"].tolist() == ids, "Geometry cache order differs")
            STUDY.require(np.array_equal(loaded["features"][:, 0], features[:, 0]), "Evaluation image features differ")
            features = loaded["features"].copy()
    features = torch.from_numpy(features)
    entries = {e["sample_id"]: e for e in protocol["development_entries"]}
    index = {sid: i for i, sid in enumerate(ids)}
    cohort = {sid for sid, observed in zip(ids, present) if observed}
    STUDY.require(len(cohort) == 75, "Photo cohort differs")
    contract = dict(kind=kind, augmentation=augmentation, fraction=fraction, source_sha256=STUDY.digest(__file__),
                    protocol_sha256=STUDY.digest(args.input / "development_nested_cv_v3.json"),
                    baseline_cache_sha256=STUDY.digest(args.cache / "baseline.npz"),
                    geometry_cache_sha256=STUDY.digest(args.features / "geometry.npz"),
                    source_groups="frozen conservative components; not verified independent people",
                    base_seed=42, subset_seed=42, lr=.001, max_inner_epochs=50, patience=5,
                    decision="fixed_median_0.5", class_weighting="training-only balanced", old_test_new_inference=False)
    freeze = root / "freeze.json"
    if freeze.exists():
        STUDY.require(json.loads(freeze.read_text()) == contract, "Image contract changed")
    else:
        STUDY.write(freeze, contract)
    rows = []
    for fold in protocol["folds"]:
        roles = {k: [sid for sid in fold[k]["sample_ids"] if sid in cohort]
                 for k in ("inner_train", "inner_val", "outer_train", "outer_holdout")}
        chosen_inner = subset(roles["inner_train"], fraction, entries)
        chosen_outer = subset(roles["outer_train"], fraction, entries)
        selected = {}
        for role, train_ids, val_ids, epoch, seed in (
            ("inner", chosen_inner, roles["inner_val"], 50, 42 + fold["fold"]),
            ("refit", chosen_outer, None, None, 142 + fold["fold"]),
        ):
            if role == "refit":
                epoch = selected["inner"]["selected_epoch"]
            identity = dict(contract=contract, fold=fold["fold"], role=role,
                            training_ids=train_ids, validation_ids=val_ids or [],
                            training_group_count=len({entries[s]["cv_component"] for s in train_ids}))
            pool = roles["inner_train" if role == "inner" else "outer_train"]
            selected[role] = train(features, labels, torch.tensor([index[s] for s in train_ids]), len(pool),
                                   torch.tensor([index[s] for s in val_ids]) if val_ids is not None else None,
                                   kind, augmentation, root / f"fold{fold['fold']}" / role, seed, epoch, identity)
        artifact = torch.load(root / f"fold{fold['fold']}/refit/classifier.pt", map_location="cpu", weights_only=True)
        model = Classifier(kind).cuda()
        model.load_state_dict(artifact["state"], strict=True)
        held = roles["outer_holdout"]
        metrics, p = score(model, features, labels, torch.tensor([index[s] for s in held]))
        fold_rows = [dict(sample_id=sid, label=int(labels[index[sid]]), probabilities=probabilities,
                          status="predicted", fold=fold["fold"]) for sid, probabilities in zip(held, p.tolist())]
        rows.extend(fold_rows)
        STUDY.write(root / f"fold{fold['fold']}/outer_predictions.json", dict(records=fold_rows, metrics=metrics))
        del model
        torch.cuda.empty_cache()
    STUDY.require(len(rows) == len({r["sample_id"] for r in rows}) == 75, "Image OOF cohort differs")
    result = dict(contract=contract, records=rows, metrics=STUDY.summarize(rows),
                  scope="same 75 image-present development records; not full 117 multimodal comparison")
    STUDY.write(root / "oof_results.json", result)
    print(json.dumps(dict(event="image_variant_complete", name=name, metrics=result["metrics"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "cache", "features", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("paired", "curves"), default="paired")
    parser.add_argument("--seconds", type=float, default=7200)
    args = parser.parse_args()
    STUDY.DEADLINE = time.monotonic() + args.seconds
    STUDY.require(args.output.resolve().is_relative_to(Path("/home/zjkj/24334067")), "Output outside personal workspace")
    torch.set_num_threads(2)
    STUDY.check_gpu()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.input / "development_nested_cv_v3.json").read_text())
    STUDY.require(STUDY.digest(args.input / "sysu_samples.csv") == protocol["csv_sha256"], "Data changed")
    meta = json.loads((args.cache / "metadata.json").read_text())
    STUDY.require(STUDY.digest(args.cache / "baseline.npz") == meta["cache_sha256"]["baseline"], "Input cache changed")
    receipt = json.loads((args.features / "receipt.json").read_text())
    STUDY.require(STUDY.digest(args.features / "geometry.npz") == receipt["files_sha256"]["geometry.npz"], "Geometry cache changed")
    summary = {}
    conditions = [(kind, aug, 1.) for kind in ("linear", "cosine") for aug in ("none", "geometry")]
    if args.phase == "curves":
        conditions = [(kind, "none", fraction) for kind in ("linear", "cosine") for fraction in (.25, .5, 1.)]
    for kind, aug, fraction in conditions:
        result = variant(args, protocol, kind, aug, fraction)
        summary[f"{kind}_{aug}_{fraction}"] = result["metrics"]
        STUDY.write(args.output / f"summary-{args.phase}.json", summary)
    STUDY.write(args.output / f"completed-{args.phase}.json", dict(completed=True, conditions=conditions, old_test_new_inference=False))


if __name__ == "__main__":
    main()
