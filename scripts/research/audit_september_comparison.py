"""Independently replay all saved feature classifiers and count errors on CPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from opmd_pred.model.fusion import MultimodalFusion, NumericEncoder, OrdinalHead

NAMES = ("image", "history", "tct", "dna", "methylation")
MISSING = {"", "NAN", "nan", "NA", "N/A", "null", "NULL", "/", "#VALUE!"}
LABELS = {"level1_mild_or_normal": 0, "level2_moderate": 1, "level3_severe": 2}
HISTORY = ["是否吸烟", "吸烟年限（年）", "吸烟量（支/天）", "是否饮酒", "饮酒年限(年)", "饮酒量(ml/天)", "是否嚼槟榔", "嚼槟榔年限(年)", "嚼槟榔量（颗/天）"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    value = value.strip()
    if value in MISSING:
        return None
    if value in ("是", "有", "yes", "Yes", "Y"):
        return 1.
    if value in ("否", "无", "no", "No", "N"):
        return 0.
    result = float(value)
    assert math.isfinite(result)
    return result


def numeric(values, scales):
    return (torch.tensor([[0. if x is None else x / s for x, s in zip(values, scales)]], dtype=torch.float32),
            torch.tensor([[x is not None for x in values]], dtype=torch.bool))


class FeatureModel(nn.Module):
    def __init__(self, config, fit):
        super().__init__()
        d = config["dimension"]
        self.fusion = nn.Module()
        f = self.fusion
        f.names = NAMES
        f.image_projection = nn.Linear(768 if config["encoder"] == "vit" else 1536, d)
        f.history_encoder = NumericEncoder(9, d)
        f.dna_encoder = NumericEncoder(config.get("dna_input_size", 2), d)
        f.methylation_encoder = NumericEncoder(1, d)
        f.tct_encoder = nn.Embedding(max(2, max(fit["tct_vocab"].values(), default=1) + 1), d)
        f.missing_tokens = nn.ParameterDict({name: nn.Parameter(torch.zeros(d)) for name in NAMES})
        f.cls_token = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=d * 2, dropout=.1, batch_first=True, norm_first=True)
        f.transformer = nn.TransformerEncoder(layer, num_layers=2)
        f.norm = nn.LayerNorm(d)
        self.ordinal_head = OrdinalHead(d) if config["head"] == "ordinal" else nn.Linear(d, 3)
        self.config = config

    def forward(self, batch):
        f = self.fusion
        embeddings = {"image": f.image_projection(batch["image_features"]), "tct": f.tct_encoder(batch["tct"])}
        present = {"image": batch["image_present"], "tct": batch["tct_present"]}
        for name in NAMES[1:]:
            if name == "tct":
                continue
            embeddings[name] = getattr(f, name + "_encoder")(batch[name], batch[name + "_mask"])
            present[name] = batch[name + "_mask"].any(1)
        output = self.ordinal_head(MultimodalFusion.fuse_embeddings(f, embeddings, present))
        if self.config["head"] == "ordinal":
            q = output.sigmoid()
            return torch.stack([1 - q[:, 0], q[:, 0] - q[:, 1], q[:, 1]], dim=1)
        return output.softmax(1)


def count(rows):
    cm = [[0] * 3 for _ in range(3)]
    assigned = len(rows)
    predicted = 0
    for row in rows:
        if row["probabilities"] is None:
            assert row["status"] == "abstain"
            continue
        assert row["status"] == "predicted"
        p = row["probabilities"]
        assert len(p) == 3 and all(math.isfinite(x) and x >= -1e-7 for x in p)
        assert abs(sum(p) - 1.) <= 1e-6
        decision = int(p[1] + p[2] > .5) + int(p[2] > .5)
        cm[row["label"]][decision] += 1
        predicted += 1
    correct = sum(cm[i][i] for i in range(3))
    return dict(correct_count=correct, assigned_count=assigned, eligible_count=predicted, abstained_count=assigned - predicted,
                accuracy_all_assigned=correct / assigned, severe_to_mild_count=cm[2][0],
                two_level_error_count=cm[2][0] + cm[0][2], confusion_matrix=cm,
                mae=sum(abs(i - j) * cm[i][j] for i in range(3) for j in range(3)) / predicted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "input", "cache", "old-csv", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    protocol = json.loads((args.input / "development_nested_cv_v3.json").read_text())
    entries = {e["sample_id"]: e for e in protocol["development_entries"]}
    locked = set(protocol["locked_previous_test_ids"] + protocol["additional_related_locked_ids"])
    assert len(entries) == 117 and not set(entries).intersection(locked)
    current = {r["sample_id"]: r for r in csv.DictReader((args.input / "sysu_samples.csv").open(encoding="utf-8-sig"))}
    old = {r["sample_id"]: r for r in csv.DictReader(args.old_csv.open(encoding="utf-8-sig"))}
    with np.load(args.cache / "baseline.npz", allow_pickle=False) as cache:
        ids, image_features, image_present = cache["sample_ids"].tolist(), cache["features"].copy(), cache["image_present"].copy()
    index = {sid: i for i, sid in enumerate(ids)}
    with np.load(args.root / "features/vit.npz", allow_pickle=False) as cache:
        vit = cache["features"].copy()
        assert cache["sample_ids"].tolist() == ids
    with np.load(args.root / "features/geometry.npz", allow_pickle=False) as cache:
        geometry = cache["features"].copy()
        assert np.array_equal(geometry[:, 0], image_features[:, 0])
    meta = json.loads((args.cache / "metadata.json").read_text())
    image_by_sha = {meta["source_image_sha_by_id"][sid]: image_features[i, 0] for i, sid in enumerate(ids) if image_present[i]}
    data_receipt = json.loads((args.root / "old-data-diagnostic-receipt.json").read_text())
    assert sha(args.old_csv) == data_receipt["old_csv_sha256"]

    def batch(sid, fit, config):
        row = old[sid] if config.get("data_version") else current[sid]
        history, hmask = numeric([number(row[k]) for k in HISTORY], [1, 50, 100, 1, 50, 1000, 1, 20, 50])
        ratio = number(row["DNA倍体分析结果"])
        threshold = number(current[sid]["dna_threshold"]) if ratio is not None else None
        dna_values = [ratio, None if threshold is None else float(threshold == 2.5)]
        if config.get("dna_input_size") == 1:
            dna_values = dna_values[:1]
        dna, dmask = numeric(dna_values, [1] * len(dna_values))
        if config["dna_scale"] == "log_standardized" and ratio is not None:
            dna[0, 0] = (math.log1p(ratio * 10000) - fit["dna_log_mean"]) / fit["dna_log_population_std"]
        methyl, mmask = numeric([number(row["甲基化总数"])], [10])
        tct = row["TCT意见"].strip()
        if tct in MISSING:
            tct = ""
        present = bool(row["image_path"])
        features = vit[index[sid], 0] if config["encoder"] == "vit" else image_features[index[sid], 0]
        if config.get("data_version") and present:
            features = image_by_sha[data_receipt["old_image_sha_by_id"][sid]]
        eligible = present or bool(hmask.any()) or bool(dmask.any()) or bool(mmask.any()) or bool(tct)
        return dict(history=history, history_mask=hmask, dna=dna, dna_mask=dmask,
                    methylation=methyl, methylation_mask=mmask, image_features=torch.from_numpy(features.copy())[None],
                    image_present=torch.tensor([present]), tct=torch.tensor([fit["tct_vocab"].get(tct, 1) if tct else 0]),
                    tct_present=torch.tensor([bool(tct)])), eligible

    checks = []
    max_error = 0.
    classifier_count = 0
    replayed_predictions = 0
    for result_path in sorted((args.root / "runs").glob("*/oof_results.json")):
        result = json.loads(result_path.read_text())
        config = result["config"]
        rows = result["records"]
        assert len(rows) == len({r["sample_id"] for r in rows}) == 117
        assert {r["sample_id"] for r in rows} == set(entries)
        for r in rows:
            assert r["label"] == LABELS[current[r["sample_id"]]["label"]]
        observed = count(rows)
        for key, value in observed.items():
            if isinstance(value, float):
                assert abs(value - result["metrics"][key]) < 1e-6, (result_path, key)
            else:
                assert value == result["metrics"][key], (result_path, key)
        variant_error = 0.
        for fold in protocol["folds"]:
            folder = result_path.parent / f"fold{fold['fold']}"
            archived = json.loads((folder / "outer_predictions.json").read_text())
            checkpoint = folder / "refit/classifier.pt"
            assert sha(checkpoint) == archived["classifier_sha256"]
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            fit = saved["fit"]
            assert set(fit["fit_ids"]) <= set(fold["outer_train"]["sample_ids"])
            assert not set(fit["fit_ids"]).intersection(fold["outer_holdout"]["sample_ids"])
            row_source = old if config.get("data_version") else current
            vocab = {t: i for i, t in enumerate(sorted({row_source[s]["TCT意见"].strip() for s in fit["fit_ids"] if row_source[s]["TCT意见"].strip() not in MISSING}), start=2)}
            assert vocab == fit["tct_vocab"]
            inner = json.loads((folder / "inner/done.json").read_text())
            assert set(inner["fit"]["fit_ids"]) <= set(fold["inner_train"]["sample_ids"])
            assert not set(inner["fit"]["fit_ids"]).intersection(fold["inner_val"]["sample_ids"])
            key = lambda x: (x["inner_validation"]["fixed_median_0.5"]["accuracy_all_assigned"], x["inner_validation"]["fixed_median_0.5"]["macro_f1"])
            assert max(inner["history"], key=key)["epoch"] == inner["selected_epoch"] == saved["selected_epoch"]
            assert len(inner["contrastive_history"]) == config["pretrain_epochs"]
            assert not any("demo" in k for k in saved["state"])
            model = FeatureModel(config, fit).eval()
            model.load_state_dict(saved["state"], strict=True)
            with torch.inference_mode():
                for row in archived["records"]:
                    item, eligible = batch(row["sample_id"], fit, config)
                    assert eligible == (row["status"] == "predicted")
                    if not eligible:
                        continue
                    actual = model(item)[0].tolist()
                    error = max(abs(a - b) for a, b in zip(actual, row["probabilities"]))
                    assert error <= 1e-4, (result_path.parent.name, fold["fold"], error)
                    variant_error = max(variant_error, error)
                    replayed_predictions += 1
            classifier_count += 1
            del model
        max_error = max(max_error, variant_error)
        checks.append(dict(name=result_path.parent.name, metrics=observed, max_replay_probability_error=variant_error))

    image_checks = []
    for result_path in sorted((args.root / "small-sample").glob("*/oof_results.json")):
        result = json.loads(result_path.read_text())
        contract = result["contract"]
        assert len(result["records"]) == len({r["sample_id"] for r in result["records"]}) == 75
        assert {r["sample_id"] for r in result["records"]} == {s for s in ids if image_present[index[s]]}
        observed = count(result["records"])
        assert observed["correct_count"] == result["metrics"]["correct_count"]
        assert observed["severe_to_mild_count"] == result["metrics"]["severe_to_mild_count"]
        for fold in protocol["folds"]:
            folder = result_path.parent / f"fold{fold['fold']}"
            saved = torch.load(folder / "refit/classifier.pt", map_location="cpu", weights_only=True)
            done = json.loads((folder / "refit/done.json").read_text())
            assert sha(folder / "refit/classifier.pt") == done["classifier_sha256"]
            train = set(saved["identity"]["training_ids"])
            assert train <= set(fold["outer_train"]["sample_ids"])
            assert not train.intersection(fold["outer_holdout"]["sample_ids"])
            inner = json.loads((folder / "inner/done.json").read_text())
            assert set(inner["identity"]["training_ids"]) <= set(fold["inner_train"]["sample_ids"])
            best = max(inner["history"], key=lambda x: (x["inner_validation"]["fixed_median_0.5"]["accuracy_all_assigned"], x["inner_validation"]["fixed_median_0.5"]["macro_f1"]))
            assert best["epoch"] == inner["selected_epoch"] == saved["selected_epoch"]
            archived = json.loads((folder / "outer_predictions.json").read_text())
            held = [r["sample_id"] for r in archived["records"]]
            features = geometry if contract["augmentation"] == "geometry" else image_features
            z = torch.from_numpy(features[[index[s] for s in held], 0])
            state = saved["state"]
            if contract["kind"] == "linear":
                logits = F.linear(z, state["weight"], state["bias"])
            else:
                logits = F.normalize(z, dim=1) @ F.normalize(state["weight"], dim=1).T * state["log_scale"].exp().clamp(max=100)
            actual = logits.softmax(1).tolist()
            for row, p in zip(archived["records"], actual):
                error = max(abs(a - b) for a, b in zip(p, row["probabilities"]))
                assert error <= 1e-4, (result_path, error)
                max_error = max(max_error, error)
                replayed_predictions += 1
            classifier_count += 1
        image_checks.append(dict(name=result_path.parent.name, metrics=observed))
    report = dict(passed=True, source_sha256=sha(__file__), model_variants=len(checks), image_conditions=len(image_checks),
                  classifiers_replayed=classifier_count, predictions_replayed=replayed_predictions,
                  max_absolute_probability_error=max_error, threshold=1e-4, counts_derived_independently=True,
                  train_validation_scope_checked=True, age_sex_inputs_absent=True, old_test_new_inference=False,
                  models=checks, image_models=image_checks, method="CPU feature-only model, strict state restoration, direct field assembly, independent error counts")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("models", "image_models")}), flush=True)


if __name__ == "__main__":
    main()
