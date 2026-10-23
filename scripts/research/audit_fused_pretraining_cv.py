"""Independent CPU replay and input/sampling audit for fused pretraining."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import warnings

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location("independent_feature_audit", Path(__file__).with_name("audit_september_comparison.py"))
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "input", "cache", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    protocol, frozen = read(args.input / "development_nested_cv_v3.json"), read(args.root / "protocol.json")
    assert A.sha(args.input / "sysu_samples.csv") == protocol["csv_sha256"] == frozen["csv_sha256"]
    assert A.sha(args.input / "development_nested_cv_v3.json") == frozen["protocol_sha256"]
    assert A.sha(args.cache / "baseline.npz") == frozen["baseline_cache_sha256"]
    entries = {entry["sample_id"]: entry for entry in protocol["development_entries"]}
    locked = set(protocol["locked_previous_test_ids"] + protocol["additional_related_locked_ids"])
    assert len(entries) == 117 and not set(entries) & locked
    with (args.input / "sysu_samples.csv").open(encoding="utf-8-sig", newline="") as stream:
        raw = {row["sample_id"]: row for row in csv.DictReader(stream) if row["sample_id"] in entries}
    with np.load(args.cache / "baseline.npz", allow_pickle=False) as cache:
        ids, features, image_present = cache["sample_ids"].tolist(), cache["features"].copy(), cache["image_present"].copy()
    assert set(ids) == set(entries) and len(ids) == 117
    index = {sid: i for i, sid in enumerate(ids)}

    def assemble(sid, fit):
        row = raw[sid]
        history, hmask = A.numeric([A.number(row[key]) for key in A.HISTORY], [1, 50, 100, 1, 50, 1000, 1, 20, 50])
        ratio, threshold = A.number(row["DNA倍体分析结果"]), A.number(row["dna_threshold"])
        assert (ratio is None) == (threshold is None)
        dna, dmask = A.numeric([ratio, None if threshold is None else float(threshold == 2.5)], [1, 1])
        methylation, mmask = A.numeric([A.number(row["甲基化总数"])], [10])
        text = row["TCT意见"].strip()
        tct = "" if text in A.MISSING else text
        present = bool(image_present[index[sid]])
        batch = dict(history=history, history_mask=hmask, dna=dna, dna_mask=dmask,
            methylation=methylation, methylation_mask=mmask,
            image_features=torch.from_numpy(features[index[sid], 0].copy())[None],
            image_present=torch.tensor([present]),
            tct=torch.tensor([fit["tct_vocab"].get(tct, 1) if tct else 0]), tct_present=torch.tensor([bool(tct)]))
        observed = [present, bool(hmask.any()), bool(tct), bool(dmask.any()), bool(mmask.any())]
        return batch, observed

    def validate_fit(fit, train_ids, held_ids):
        eligible = [sid for sid in train_ids if any(assemble(sid, fit)[1])]
        assert fit["fit_ids"] == eligible and not set(eligible) & set(held_ids)
        vocabulary = sorted({raw[sid]["TCT意见"].strip() for sid in eligible if raw[sid]["TCT意见"].strip() not in A.MISSING})
        assert fit["tct_vocab"] == {text: i for i, text in enumerate(vocabulary, start=2)}
        assert fit["class_counts"] == [sum(A.LABELS[raw[sid]["label"]] == label for sid in eligible) for label in range(3)]

    replayed, maximum, outer_checks = 0, 0., []
    candidates = sorted((args.root / "runs").glob("*/oof_results.json"))
    expected = {(mode, seed) for mode in frozen["arguments"]["pretrain_mode"] for seed in (42, 43, 44)}
    assert {(read(path)["method"], read(path)["seed"]) for path in candidates} == expected
    references = [args.reference / "runs" / f"{method}_seed{seed}" / "oof_results.json"
                  for method in ("ordinal", "ordinal_pretrain") for seed in (42, 43, 44)]
    for path in candidates + references:
        result, all_rows = read(path), []
        config = result["config"]
        assert config["head"] == "ordinal" and config["balanced"] is False and config["encoder"] == "dino"
        variant_maximum = 0.
        for fold in protocol["folds"]:
            folder = path.parent / f"fold{fold['fold']}"
            saved = read(folder / "outer_predictions.json")
            artifact_path = folder / "refit/classifier.pt"
            assert A.sha(artifact_path) == saved["classifier_sha256"]
            artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
            train, held = fold["outer_train"]["sample_ids"], fold["outer_holdout"]["sample_ids"]
            assert artifact["contract"]["train_ids"] == train and artifact["contract"]["validation_ids"] is None
            validate_fit(artifact["fit"], train, held)
            for field in ("cv_component", "image_sha256", "dna_report_sha256"):
                assert not {entries[sid][field] for sid in train if entries[sid][field]} & {entries[sid][field] for sid in held if entries[sid][field]}
            model = A.FeatureModel(config, artifact["fit"]).eval()
            model.load_state_dict(artifact["state"], strict=True)
            assert not any("projector" in key or "demographics" in key for key in artifact["state"])
            assert [row["sample_id"] for row in saved["records"]] == held
            with torch.inference_mode():
                for row in saved["records"]:
                    sid = row["sample_id"]
                    batch, observed = assemble(sid, artifact["fit"])
                    assert row["label"] == A.LABELS[raw[sid]["label"]]
                    assert row["status"] == ("predicted" if any(observed) else "abstain")
                    if any(observed):
                        probability = model(batch)[0]
                        error = max(abs(a - b) for a, b in zip(probability.tolist(), row["probabilities"], strict=True))
                        assert error <= 1e-4
                        variant_maximum = max(variant_maximum, error)
                        replayed += 1
                    else:
                        assert row["probabilities"] is None
            all_rows.extend(saved["records"])
        assert all_rows == result["records"]
        counted = A.count(all_rows)
        assert counted["assigned_count"] == 117 and counted["abstained_count"] == 5
        for key, value in counted.items():
            assert abs(value - result["metrics"][key]) < 1e-6 if isinstance(value, float) else value == result["metrics"][key]
        outer_checks.append(dict(method=result["method"], seed=result["seed"], metrics=counted, max_probability_error=variant_maximum))
        maximum = max(maximum, variant_maximum)

    stage_checks, initial_checks, batches_checked = 0, 0, 0
    common_schedules, common_bases = {}, {}
    for path in (args.root / "runs").glob("*/*/*/done.json"):
        done = read(path)
        checkpoint = torch.load(path.parent / "classifier.pt", map_location="cpu", weights_only=True)
        assert A.sha(path.parent / "classifier.pt") == done["classifier_sha256"]
        assert checkpoint["contract"] == done["contract"] and checkpoint["selected_epoch"] == done["selected_epoch"]
        if done["contract"]["validation_ids"] is not None:
            best = max(done["history"], key=lambda item: (item["inner_validation"]["fixed_median_0.5"]["accuracy_all_assigned"],
                       item["inner_validation"]["fixed_median_0.5"]["macro_f1"], -item["epoch"]))
            assert best["epoch"] == done["selected_epoch"] and len(done["history"]) <= 50
        stage_checks += 1
    for path in (args.root / "runs").glob("*/*/*-initial/initial.pt"):
        initial = torch.load(path, map_location="cpu", weights_only=True)
        identity, fit = initial["identity"], initial["fit"]
        mode = identity["config"]["pretrain_mode"]
        variant = path.parents[2].name
        seed = int(variant.rsplit("seed", 1)[1])
        fold = int(path.parents[1].name.removeprefix("fold"))
        role = identity["role"]
        key = (seed, fold, role)
        shared_path = args.root / "shared" / f"seed{seed}" / f"fold{fold}" / role / "base.pt"
        base = torch.load(shared_path, map_location="cpu", weights_only=True)
        archived_base = torch.load(args.reference / "runs" / f"ordinal_seed{seed}" / f"fold{fold}" /
                                   f"{role}-initial" / "initial.pt", map_location="cpu", weights_only=True)
        assert A.sha(shared_path) == initial["base_sha256"]
        assert common_bases.setdefault(key, initial["base_sha256"]) == initial["base_sha256"]
        assert base["archived_no_pretrain_initial_equal"] is True
        assert base["fit"] == archived_base["fit"]
        assert base["state"].keys() == archived_base["state"].keys()
        assert all(torch.equal(value, archived_base["state"][name]) for name, value in base["state"].items())
        assert torch.equal(base["cpu_rng"], archived_base["cpu_rng"]) and torch.equal(base["cuda_rng"], archived_base["cuda_rng"])
        assert torch.equal(initial["cpu_rng"], base["cpu_rng"]) and torch.equal(initial["cuda_rng"], base["cuda_rng"])
        assert set(identity["train_ids"]) <= set(entries) and not set(identity["train_ids"]) & locked
        validate_fit(fit, identity["train_ids"], [])
        expected_ids = [sid for sid in identity["train_ids"] if sum(assemble(sid, fit)[1]) >= 2]
        assert initial["pretrain_ids"] == expected_ids
        assert len(initial["pretraining_history"]) == 30
        schedule, exposure = [], {name: 0 for name in A.NAMES}
        updates = 0
        for epoch in initial["pretraining_history"]:
            used = []
            for batch in epoch["batches"]:
                used.extend(batch["sample_ids"])
                if batch.get("skipped_single_record_batch"):
                    assert len(batch["sample_ids"]) == 1
                    continue
                labels = [A.LABELS[raw[sid]["label"]] for sid in batch["sample_ids"]]
                assert labels == batch["labels"]
                assert set(batch["sample_ids"]) <= set(expected_ids)
                actual_masks = [assemble(sid, fit)[1] for sid in batch["sample_ids"]]
                assert actual_masks == batch["observed"]
                for observed, retained in zip(batch["observed"], batch["retained_b"], strict=True):
                    assert sum(observed) >= 2 and sum(retained) >= 2
                    assert all(not keep or available for keep, available in zip(retained, observed, strict=True))
                    if mode == "fused_subset_class" and sum(observed) > 2:
                        assert sum(retained) < sum(observed)
                    else:
                        assert retained == observed
                    for column, name in enumerate(A.NAMES):
                        exposure[name] += int(not observed[column]) + int(not retained[column])
                assert all(1 <= a <= 4 and 1 <= b <= 4 and a != b for a, b in zip(batch["image_views_a"], batch["image_views_b"], strict=True))
                positive = sum(a == b for a in labels for b in labels) if mode != "fused_full_instance" else len(labels)
                assert batch["positive_pairs"] == positive and batch["negative_pairs"] == len(labels) ** 2 - positive
                assert all(math.isfinite(value) and value >= 0 for value in batch["gradient_norms"].values())
                schedule.append([epoch["epoch"], batch["sample_ids"], batch["image_views_a"], batch["image_views_b"], batch["observed"]])
                updates += 1
                batches_checked += 1
            assert set(used) == set(expected_ids) and len(used) == len(expected_ids)
        assert updates == initial["pretraining_updates"] and exposure == initial["missing_exposures"]
        signature = hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest()
        assert common_schedules.setdefault(key, signature) == signature, "Candidate modes used different record/image schedules"
        assert initial["gradients"]["fusion"] > 0 and initial["gradients"]["projector"] > 0
        for name in A.NAMES:
            if exposure[name]:
                assert initial["gradients"]["missing." + name] > 0
        for prefix, group in (("fusion.transformer.", "fusion"), ("fusion.missing_tokens.", "missing_tokens")):
            delta = math.sqrt(sum(float((value - base["state"][name]).square().sum())
                                 for name, value in initial["state"].items() if name.startswith(prefix)))
            assert delta > 0 and abs(delta - initial["parameter_delta_norms"][group]) < 1e-7
        assert all(torch.equal(value, base["state"][name]) for name, value in initial["state"].items() if name.startswith("ordinal_head."))
        initial_checks += 1
    assert len(candidates) == 9 and stage_checks == initial_checks == 90
    assert len(common_schedules) == len(common_bases) == 30
    report = dict(passed=True, new_variants=9, reference_variants=6, strict_outer_classifiers=75,
        training_stages_checked=stage_checks, pretraining_stages_checked=initial_checks,
        batches_checked=batches_checked, shared_record_and_image_schedules=30,
        replayed_probability_vectors=replayed, maximum_probability_error=maximum,
        no_single_modality_contrastive_views=True, no_locked_test_inputs=True,
        fusion_and_exposed_missing_token_gradients_verified=True, checks=outer_checks)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}))


if __name__ == "__main__":
    main()
