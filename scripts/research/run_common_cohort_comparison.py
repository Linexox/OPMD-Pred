"""Freeze a shared image/DNA cohort, then fit matched nested linear baselines.

Patient-level inputs, predictions and fitted heads stay in --private-output.
No image encoder forward pass, new split, old-test evaluation or APP update.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


LABELS = {"level1_mild_or_normal": 0, "level2_moderate": 1, "level3_severe": 2}
ENCODER_SHA = "58a3fcc724155c8ecfc7bf0e955b2bf022f0e2aa4bfed06a51a5b99c850a4a65"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def median(probabilities):
    return ((probabilities[:, 1:].sum(axis=1) > 0.5).astype(np.int64)
            + (probabilities[:, 2] > 0.5).astype(np.int64))


def metrics(labels, predictions):
    cm = np.zeros((3, 3), dtype=int)
    for actual, predicted in zip(labels, predictions):
        cm[int(actual), int(predicted)] += 1
    return {"n": len(labels), "correct": int(np.trace(cm)),
            "accuracy": float(np.trace(cm) / len(labels)),
            "macro_f1": float(f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
            "severe_to_mild": int(cm[2, 0]), "severe_to_moderate": int(cm[2, 1]),
            "cross_two": int(cm[2, 0] + cm[0, 2]),
            "severe_recall": float(cm[2, 2] / cm[2].sum()),
            "mae": float(np.abs(labels - predictions).mean()), "confusion": cm.tolist()}


def prepare(args, protocol):
    cv, cache_meta = load(args.cv_protocol), load(args.cache / "metadata.json")
    require(sha(args.csv) == cv["csv_sha256"] == cache_meta["csv_sha256"], "CSV fingerprint mismatch")
    require(sha(args.cv_protocol) == cache_meta["cv_protocol_sha256"], "CV fingerprint mismatch")
    require(cache_meta["encoder_sha256"] == ENCODER_SHA, "Wrong frozen encoder")
    require(sha(args.cache / "baseline.npz") == cache_meta["cache_sha256"]["baseline"], "Feature cache changed")
    require(cache_meta["evaluation_view"] == 0 and cache_meta["previous_test_inference"] is False,
            "Wrong feature view or old test inference receipt")
    expected = {row["sample_id"]: row for row in cv["development_entries"]}
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        source_rows = {row["sample_id"]: row for row in csv.DictReader(stream)}
    cohort = []
    for sid, entry in expected.items():
        row = source_rows[sid]
        if row["image_path"].strip() and row["DNA倍体分析结果"].strip():
            require(row["dna_threshold"] and float(row["dna_threshold"]) in (2.3, 2.5), "DNA definition incomplete")
            require(np.isfinite(float(row["DNA倍体分析结果"])) and float(row["DNA倍体分析结果"]) >= 0,
                    "Invalid DNA ratio")
            require(LABELS[row["label"]] == entry["label"], "Label differs from split")
            cohort.append(sid)
    support = [sum(expected[s]["label"] == i for s in cohort) for i in range(3)]
    require(len(cohort) == protocol["expected_count"] and support == protocol["expected_support"],
            "Unexpected common-cohort population")
    require(not set(cohort) & set(cv["locked_previous_test_ids"]), "Old test included")
    subset, folds = set(cohort), []
    for fold in cv["folds"]:
        item = {"fold": fold["fold"]}
        for role in ("outer_train", "outer_holdout", "inner_train", "inner_val"):
            item[role] = [sid for sid in fold[role]["sample_ids"] if sid in subset]
            require(bool(item[role]), "Empty common-cohort split")
        require(set(item["outer_train"]) | set(item["outer_holdout"]) == subset,
                "Outer split no longer covers shared cohort")
        require(not set(item["outer_train"]) & set(item["outer_holdout"]), "Outer overlap")
        require(set(item["inner_train"]) | set(item["inner_val"]) == set(item["outer_train"]), "Inner split differs")
        for left, right in (("outer_train", "outer_holdout"), ("inner_train", "inner_val")):
            require(not {expected[s]["cv_component"] for s in item[left]}
                    & {expected[s]["cv_component"] for s in item[right]}, "Source group crosses split")
        require({expected[s]["label"] for s in item["inner_train"]} == {0, 1, 2}, "Inner train lacks class")
        item["support"] = {role: [sum(expected[s]["label"] == i for s in item[role]) for i in range(3)]
                           for role in ("outer_train", "outer_holdout", "inner_train", "inner_val")}
        folds.append(item)
    held_ids = [s for f in folds for s in f["outer_holdout"]]
    require(len(held_ids) == len(set(held_ids)) == len(cohort) and set(held_ids) == subset, "Wrong OOF cover")
    manifest = {"study_id": protocol["study_id"], "csv_sha256": sha(args.csv),
                "cv_sha256": sha(args.cv_protocol), "cache_sha256": sha(args.cache / "baseline.npz"),
                "metadata_sha256": sha(args.cache / "metadata.json"), "protocol_sha256": sha(args.protocol),
                "encoder_sha256": ENCODER_SHA, "count": len(cohort), "support": support,
                "sample_ids": cohort, "source_components": len({expected[s]["cv_component"] for s in cohort}),
                "folds": folds, "old_test_overlap": 0, "per_record_prebiopsy_timing_verified": False}
    return manifest, source_rows


def fit_model(X, y, indices, C, settings):
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=C, **settings))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(X[indices], y[indices])
    require(pipeline[-1].classes_.tolist() == [0, 1, 2], "Classifier classes differ")
    return pipeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("csv", "cv-protocol", "cache", "protocol", "cohort-manifest", "private-output", "summary"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    protocol = load(args.protocol)
    manifest, source = prepare(args, protocol)
    if args.prepare_only:
        require(not args.cohort_manifest.exists(), "Never overwrite frozen cohort")
        args.cohort_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.cohort_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"frozen_count": manifest["count"], "support": manifest["support"],
                          "fold_sizes": [len(f["outer_holdout"]) for f in manifest["folds"]],
                          "manifest_sha256": sha(args.cohort_manifest)}))
        return
    require(load(args.cohort_manifest) == manifest, "Prepared cohort or protocol changed")
    require(not args.private_output.exists() and not args.summary.exists(), "Never overwrite previous result")
    started = time.monotonic()
    ids = manifest["sample_ids"]
    index = {sid: i for i, sid in enumerate(ids)}
    with np.load(args.cache / "baseline.npz", allow_pickle=False) as cached:
        cache_ids = cached["sample_ids"].tolist()
        require(cache_ids == [r["sample_id"] for r in load(args.cv_protocol)["development_entries"]], "Cache order changed")
        cix = [cache_ids.index(s) for s in ids]
        require(cached["image_present"][cix].all(), "Common cohort contains missing image")
        image = cached["features"][cix, 0].astype(np.float64)
        labels = cached["labels"][cix].astype(np.int64)
    require(image.shape == (manifest["count"], 1536) and np.isfinite(image).all(), "Wrong image features")
    require(labels.tolist() == [LABELS[source[s]["label"]] for s in ids], "Feature labels differ")
    dna = np.asarray([[np.log1p(10000 * float(source[s]["DNA倍体分析结果"])),
                       float(float(source[s]["dna_threshold"]) == 2.5)] for s in ids])
    matrices = {"image": image, "dna": dna, "image_dna": np.concatenate([image, dna], axis=1)}
    settings = {k: v for k, v in protocol["classifier"].items() if k != "name"}
    predictions = {name: np.empty((len(ids), 3)) for name in protocol["models"]}
    details, selected_models = [], []
    args.private_output.mkdir(parents=True)
    with threadpool_limits(limits=1):
        for fold in manifest["folds"]:
            ix = {role: np.asarray([index[s] for s in fold[role]], dtype=int)
                  for role in ("outer_train", "outer_holdout", "inner_train", "inner_val")}
            for name in protocol["models"]:
                X = matrices[name]
                trials = []
                for C in protocol["C_grid"]:
                    model = fit_model(X, labels, ix["inner_train"], C, settings)
                    result = metrics(labels[ix["inner_val"]], median(model.predict_proba(X[ix["inner_val"]])))
                    trials.append({"C": C, "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]})
                best = max(trials, key=lambda row: (row["accuracy"], row["macro_f1"], -row["C"]))
                model = fit_model(X, labels, ix["outer_train"], best["C"], settings)
                probabilities = model.predict_proba(X[ix["outer_holdout"]])
                predictions[name][ix["outer_holdout"]] = probabilities
                scaler, classifier = model[0], model[-1]
                model_path = args.private_output / f"fold{fold['fold']}-{name}.npz"
                np.savez(model_path, mean=scaler.mean_, scale=scaler.scale_, coef=classifier.coef_,
                         intercept=classifier.intercept_, classes=classifier.classes_)
                with np.load(model_path, allow_pickle=False) as saved:
                    logits = ((X[ix["outer_holdout"]] - saved["mean"]) / saved["scale"]) @ saved["coef"].T + saved["intercept"]
                    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
                    restored = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                difference = float(np.abs(restored - probabilities).max())
                require(difference < 1e-10, "Saved head does not reproduce probabilities")
                selected_models.append({"fold": fold["fold"], "model": name, "selected_C": best["C"],
                                        "inner_trials": trials, "model_sha256": sha(model_path),
                                        "max_saved_model_probability_difference": difference,
                                        "max_solver_iterations": int(classifier.n_iter_.max())})
            print(json.dumps({"fold": fold["fold"], "held_count": len(ix["outer_holdout"]), "status": "complete"}), flush=True)
    summary_metrics = {}
    for name, probabilities in predictions.items():
        require(np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1), 1), "Invalid OOF probabilities")
        predicted = median(probabilities)
        summary_metrics[name] = metrics(labels, predicted)
        details.extend({"sample_id": sid, "label": int(labels[i]), "method": name,
                        "probabilities": probabilities[i].tolist(), "prediction": int(predicted[i])}
                       for i, sid in enumerate(ids))
    image_pred, joint_pred = median(predictions["image"]), median(predictions["image_dna"])
    primary, baseline = summary_metrics["image_dna"], summary_metrics["image"]
    paired = {"wrong_to_correct": int(((image_pred != labels) & (joint_pred == labels)).sum()),
              "correct_to_wrong": int(((image_pred == labels) & (joint_pred != labels)).sum()),
              "severe_to_mild_removed": int(((labels == 2) & (image_pred == 0) & (joint_pred != 0)).sum()),
              "severe_to_mild_added": int(((labels == 2) & (image_pred != 0) & (joint_pred == 0)).sum())}
    gate = {"severe_reduction": baseline["severe_to_mild"] - primary["severe_to_mild"],
            "cross_two_increase": primary["cross_two"] - baseline["cross_two"],
            "correct_count_loss": baseline["correct"] - primary["correct"]}
    conditions = protocol["continuation_gate"]
    gate["passed"] = (gate["severe_reduction"] >= conditions["severe_to_mild_reduction_at_least"]
                      and gate["cross_two_increase"] <= conditions["cross_two_error_increase_at_most"]
                      and gate["correct_count_loss"] <= conditions["correct_count_loss_at_most"])
    (args.private_output / "oof-predictions.private.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (args.private_output / "selected-heads.private.json").write_text(json.dumps(selected_models, indent=2), encoding="utf-8")
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")}
    payload = {"study_id": protocol["study_id"], "status": "completed_exploratory_cross_sectional_comparison",
               "n": manifest["count"], "support": manifest["support"], "models": summary_metrics,
               "paired_image_to_image_dna": paired, "primary_gate": gate,
               "cohort_manifest_sha256": sha(args.cohort_manifest), "protocol_sha256": sha(args.protocol),
               "script_sha256": sha(Path(__file__)), "packages": packages,
               "private_output": str(args.private_output), "fitted_final_heads": len(selected_models),
               "max_saved_model_probability_difference": max(r["max_saved_model_probability_difference"] for r in selected_models),
               "selected_heads_sha256": sha(args.private_output / "selected-heads.private.json"),
               "oof_predictions_sha256": sha(args.private_output / "oof-predictions.private.json"),
               "elapsed_seconds": time.monotonic() - started, "new_simple_linear_heads_fitted": True,
               "new_encoder_training_or_forward": False, "APP_model_changed": False,
               "old_test22_evaluated": False, "clinical_efficacy_evaluated": False,
               "limitations": protocol["interpretation"]}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": payload["n"], "models": summary_metrics, "gate": gate}), flush=True)


if __name__ == "__main__":
    main()
