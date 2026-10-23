"""CPU reconstruction of inputs, checkpoints, epoch/LR choices and held-out counts."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location("independent_september_audit", Path(__file__).with_name("audit_september_comparison.py"))
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "input", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    protocol = read(args.input / "development_nested_cv_v3.json")
    frozen = read(args.root / "protocol.json")
    assert A.sha(args.input / "sysu_samples.csv") == protocol["csv_sha256"] == frozen["csv_sha256"]
    assert A.sha(args.input / "development_nested_cv_v3.json") == frozen["protocol_sha256"]
    assert A.sha(args.cache / "baseline.npz") == frozen["baseline_cache_sha256"]
    entries = {row["sample_id"]: row for row in protocol["development_entries"]}
    locked = set(protocol["locked_previous_test_ids"] + protocol["additional_related_locked_ids"])
    assert len(entries) == 117 and not set(entries).intersection(locked)
    raw = {r["sample_id"]: r for r in csv.DictReader((args.input / "sysu_samples.csv").open(encoding="utf-8-sig"))}
    with np.load(args.cache / "baseline.npz", allow_pickle=False) as cache:
        ids, features, image_present = cache["sample_ids"].tolist(), cache["features"].copy(), cache["image_present"].copy()
    assert set(ids) == set(entries) and len(ids) == 117
    index = {sid: i for i, sid in enumerate(ids)}

    def assemble(sid, fit):
        row = raw[sid]
        history, hmask = A.numeric([A.number(row[key]) for key in A.HISTORY], [1, 50, 100, 1, 50, 1000, 1, 20, 50])
        ratio, threshold = A.number(row["DNA倍体分析结果"]), A.number(row["dna_threshold"])
        assert (ratio is None) == (threshold is None)
        assert threshold is None or threshold in (2.3, 2.5)
        dna, dmask = A.numeric([ratio, None if threshold is None else float(threshold == 2.5)], [1, 1])
        methylation, mmask = A.numeric([A.number(row["甲基化总数"])], [10])
        tct = row["TCT意见"].strip()
        if tct in A.MISSING:
            tct = ""
        present = bool(image_present[index[sid]])
        batch = dict(history=history, history_mask=hmask, dna=dna, dna_mask=dmask,
            methylation=methylation, methylation_mask=mmask,
            image_features=torch.from_numpy(features[index[sid], 0].copy())[None],
            image_present=torch.tensor([present]),
            tct=torch.tensor([fit["tct_vocab"].get(tct, 1) if tct else 0]), tct_present=torch.tensor([bool(tct)]))
        eligible = present or bool(hmask.any()) or bool(dmask.any()) or bool(mmask.any()) or bool(tct)
        return batch, eligible

    def validate_fit(fit, train_ids, held_ids):
        assert set(fit["fit_ids"]) <= set(train_ids)
        assert not set(fit["fit_ids"]).intersection(held_ids)
        available = [sid for sid in train_ids if assemble(sid, fit)[1]]
        assert fit["fit_ids"] == available
        texts = sorted({raw[sid]["TCT意见"].strip() for sid in available
                        if raw[sid]["TCT意见"].strip() not in A.MISSING})
        assert fit["tct_vocab"] == {text: i for i, text in enumerate(texts, start=2)}
        assert fit["class_counts"] == [sum(A.LABELS[raw[sid]["label"]] == y for sid in available) for y in range(3)]

    checks, replayed, maximum = [], 0, 0.
    for path in sorted((args.root / "runs").glob("*/oof_results.json")):
        result = read(path)
        config = result["config"]
        assert config["encoder"] == "dino" and config["dimension"] == 128
        assert config["dropout"] == "bernoulli" and config["dna_scale"] == "raw"
        assert len(result["records"]) == 117 and {r["sample_id"] for r in result["records"]} == set(entries)
        variant_error, variant_rows = 0., []
        for fold in protocol["folds"]:
            folder = path.parent / f"fold{fold['fold']}"
            saved = read(folder / "outer_predictions.json")
            checkpoint_path = Path(saved.get("classifier_path", folder / "refit/classifier.pt"))
            assert A.sha(checkpoint_path) == saved["classifier_sha256"]
            artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            assert artifact["config"] == config
            held, train = fold["outer_holdout"]["sample_ids"], fold["outer_train"]["sample_ids"]
            assert artifact["contract"]["train_ids"] == train and artifact["contract"]["validation_ids"] is None
            validate_fit(artifact["fit"], train, held)
            assert not set(train).intersection(held) and set(train) | set(held) == set(entries)
            for field in ("cv_component", "image_sha256", "dna_report_sha256"):
                assert not {entries[sid][field] for sid in train if entries[sid][field]}.intersection(
                    {entries[sid][field] for sid in held if entries[sid][field]})
            model = A.FeatureModel(config, artifact["fit"]).eval()
            model.load_state_dict(artifact["state"], strict=True)
            by_id = {r["sample_id"]: r for r in saved["records"]}
            assert list(by_id) == held
            with torch.inference_mode():
                for sid in held:
                    batch, eligible = assemble(sid, artifact["fit"])
                    row = by_id[sid]
                    assert row["label"] == A.LABELS[raw[sid]["label"]]
                    assert row["status"] == ("predicted" if eligible else "abstain")
                    if eligible:
                        p = model(batch)[0]
                        assert torch.isfinite(p).all() and bool((p >= -1e-7).all())
                        assert abs(float(p.sum()) - 1) < 1e-6
                        error = max(abs(a - b) for a, b in zip(p.tolist(), row["probabilities"]))
                        variant_error = max(variant_error, error)
                        replayed += 1
                        assert error <= 1e-4
                    else:
                        assert row["probabilities"] is None
            variant_rows.extend(saved["records"])
            if result["method"] == "ordinal_pretrain_tuned":
                fixed = args.root / "runs" / f"ordinal_pretrain_seed{result['seed']}" / f"fold{fold['fold']}"
                normal, low = read(fixed / "inner/done.json"), read(folder / "inner-low/done.json")
                assert normal["initial_sha256"] == low["initial_sha256"]
                choice = max((normal, low), key=lambda r: (tuple(r["best_key"]), r["lr"], -r["selected_epoch"]))
                assert saved["selection"]["lr"] == choice["lr"] == artifact["contract"]["lr"]
                assert saved["selection"]["epoch"] == choice["selected_epoch"] == artifact["selected_epoch"]
            del model
        independent = A.count(variant_rows)
        for key, value in independent.items():
            if key == "mae":
                assert abs(value - result["metrics"][key]) < 1e-6
            else:
                assert value == result["metrics"][key], (result["method"], key)
        assert independent["abstained_count"] == 5 and independent["eligible_count"] == 112
        checks.append(dict(method=result["method"], seed=result["seed"], metrics=independent,
                           max_probability_error=variant_error))
        maximum = max(maximum, variant_error)

    stage_checks, initial_checks = 0, 0
    for done_path in (args.root / "runs").glob("*/*/*/done.json"):
        done = read(done_path)
        artifact = torch.load(done_path.parent / "classifier.pt", map_location="cpu", weights_only=True)
        assert A.sha(done_path.parent / "classifier.pt") == done["classifier_sha256"]
        assert artifact["contract"] == done["contract"]
        assert artifact["selected_epoch"] == done["selected_epoch"]
        if done["contract"]["validation_ids"] is not None:
            selection = max(done["history"], key=lambda item: (
                item["inner_validation"]["fixed_median_0.5"]["accuracy_all_assigned"],
                item["inner_validation"]["fixed_median_0.5"]["macro_f1"], -item["epoch"]))
            assert selection["epoch"] == done["selected_epoch"]
            assert len(done["history"]) <= 50
        stage_checks += 1
    for initial_path in (args.root / "runs").glob("*/*/*-initial/initial.pt"):
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        expected_epochs = initial["identity"]["config"]["pretrain_epochs"]
        assert len(initial["pretraining_history"]) == expected_epochs
        assert initial["pretraining_updates"] == sum(r["effective_updates"] for r in initial["pretraining_history"])
        assert not set(initial["identity"]["train_ids"]).intersection(locked)
        validate_fit(initial["fit"], initial["identity"]["train_ids"], [])
        if expected_epochs:
            assert initial["pretraining_updates"] > 0 and sum(r["paired_records"] for r in initial["pair_totals"].values()) > 0
        initial_checks += 1
    summary = read(args.root / "summary.json")
    assert len(checks) == 18 and len(summary) == 6
    assert all(len(item["per_seed"]) == 3 for item in summary.values())
    report = dict(passed=True, variants_checked=len(checks), selected_outer_classifiers=5 * len(checks),
        strict_checkpoint_and_stage_checks=stage_checks, initial_and_pretraining_checks=initial_checks,
        replayed_probability_vectors=replayed, maximum_probability_error=maximum, checks=checks,
        no_locked_test_inputs=True, inner_only_lr_selection=True, independent_input_assembly=True)
    write(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}))


if __name__ == "__main__":
    main()
