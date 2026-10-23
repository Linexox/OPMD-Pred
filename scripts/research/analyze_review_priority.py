"""Audit archived OOF probabilities and compare fixed-workload review priorities.

Only aggregate counts leave the private output directory. No model is fitted or
executed. Old test IDs are used only to verify exclusion from development data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path


LABELS = {"level1_mild_or_normal": 0, "level2_moderate": 1, "level3_severe": 2}
HISTORY = (
    "是否吸烟", "吸烟年限（年）", "吸烟量（支/天）", "是否饮酒", "饮酒年限(年)",
    "饮酒量(ml/天)", "是否嚼槟榔", "嚼槟榔年限(年)", "嚼槟榔量（颗/天）",
)
MISSING = {"", "NAN", "nan", "NA", "N/A", "null", "NULL", "/", "#VALUE!"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def observed(value: str) -> bool:
    return value.strip() not in MISSING


def modality_count(row: dict) -> int:
    return sum((observed(row["image_path"]), any(observed(row[k]) for k in HISTORY),
                observed(row["TCT意见"]), observed(row["DNA倍体分析结果"]),
                observed(row["甲基化总数"])))


def median_label(probabilities: list[float]) -> int:
    return int(sum(probabilities[1:]) > 0.5) + int(probabilities[2] > 0.5)


def allocate(counts: dict[int, int], budget: int) -> dict[int, int]:
    total = sum(counts.values())
    require(0 <= budget <= total, "Review budget outside eligible cohort")
    quotas = {fold: budget * count // total for fold, count in counts.items()}
    order = sorted(counts, key=lambda fold: (-(budget * counts[fold] % total), fold))
    for fold in order[:budget - sum(quotas.values())]:
        quotas[fold] += 1
    require(sum(quotas.values()) == budget, "Wrong allocated workload")
    return quotas


def counts(rows: list[dict]) -> dict:
    cm = [[0] * 3 for _ in range(3)]
    for row in rows:
        cm[row["label"]][row["prediction"]] += 1
    return {
        "n": len(rows),
        "correct": sum(cm[i][i] for i in range(3)),
        "errors": sum(row["prediction"] != row["label"] for row in rows),
        "severe_to_mild": cm[2][0],
        "severe_to_moderate": cm[2][1],
        "cross_two": cm[2][0] + cm[0][2],
        "confusion": cm,
    }


def describe_selection(rows: list[dict], selected: set[str], no_input_count: int) -> dict:
    reviewed = [r for r in rows if r["sample_id"] in selected]
    remaining = [r for r in rows if r["sample_id"] not in selected]
    return {"model_records_reviewed": len(selected),
            "mandatory_no_input_review": no_input_count,
            "total_review_workload": len(selected) + no_input_count,
            "captured": counts(reviewed), "remaining": counts(remaining),
            "correct_records_sent_for_review": counts(reviewed)["correct"],
            "human_corrections_observed": None}


def empirical(values: list[int]) -> dict:
    ordered = sorted(values)
    return {"median": statistics.median(values), "mean": statistics.mean(values),
            "empirical_2_5_percentile": ordered[math.floor(0.025 * (len(values) - 1))],
            "empirical_97_5_percentile": ordered[math.ceil(0.975 * (len(values) - 1))]}


def audit_inputs(args, protocol: dict) -> tuple[list[dict], list[dict], dict, list[dict]]:
    cv = load(args.cv_protocol)
    csv_sha, cv_sha = sha(args.csv), sha(args.cv_protocol)
    require(csv_sha == cv["csv_sha256"], "Frozen CSV fingerprint differs")
    expected = {r["sample_id"]: r["label"] for r in cv["development_entries"]}
    require(len(expected) == protocol["development_count"], "Wrong development count")
    require(not set(expected) & set(cv["locked_previous_test_ids"]), "Old test in development")
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = {r["sample_id"]: r for r in csv.DictReader(stream)}
    private_root = args.archive / f"seed{args.seed}" / "control"
    oof_path = private_root / "oof_results.json"
    oof = load(oof_path)
    archived = oof["results"]["baseline"]
    combined = {r["sample_id"]: r for r in archived["records"]}
    require(len(archived["records"]) == len(combined) == len(expected), "Duplicate/wrong OOF IDs")
    require(set(combined) == set(expected), "OOF coverage differs from frozen development")
    files = [{"path": str(p), "sha256": sha(p)} for p in (args.csv, args.cv_protocol, args.protocol, oof_path)]
    eligible, abstained, seen = [], [], []
    max_delta = 0.0
    for fold in cv["folds"]:
        index = fold["fold"]
        path = private_root / f"fold{index}" / "baseline" / "outer_predictions.json"
        document = load(path)
        ids = fold["outer_holdout"]["sample_ids"]
        require(document["fold"] == index and document["recipe"] == "baseline", "Wrong outer fold")
        require([r["sample_id"] for r in document["records"]] == ids, "Holdout IDs/order differ")
        receipt = document["receipt"]
        require(receipt["csv_sha256"] == csv_sha and receipt["cv_sha256"] == cv_sha,
                "Probability receipts do not bind current frozen inputs")
        require(receipt["distance_weight"] == 0 and receipt["seed_offset"] == args.seed - 42,
                "Wrong model method/seed")
        require(receipt["previous_test_inference"] is False, "Receipt reports old test evaluation")
        require(receipt == oof["receipt"], "Fold and combined receipts differ")
        model_path = path.parent / "refit" / "classifier.pt"
        require(sha(model_path) == document["classifier_sha256"], "Classifier bytes differ")
        files.extend({"path": str(p), "sha256": sha(p)} for p in (path, model_path))
        for item in document["records"]:
            sid = item["sample_id"]
            require(item["label"] == expected[sid] == LABELS[csv_rows[sid]["label"]], "Label mismatch")
            require(combined[sid]["label"] == expected[sid], "Combined OOF label mismatch")
            require(item["status"] == combined[sid]["status"], "OOF status mismatch")
            number = modality_count(csv_rows[sid])
            row = dict(item, fold=index, modality_count=number)
            if item["status"] == "abstain":
                require(item["probabilities"] is None and number == 0, "Invalid no-input abstention")
                require(combined[sid]["probabilities"] is None, "Combined record fabricates probabilities")
                abstained.append(row)
            else:
                p = item["probabilities"]
                require(item["status"] == "predicted" and number > 0, "Wrong eligibility")
                require(len(p) == 3 and all(math.isfinite(x) and 0 <= x <= 1 for x in p)
                        and abs(sum(p) - 1) < 1e-6, "Invalid probabilities")
                delta = max(abs(x-y) for x, y in zip(p, combined[sid]["probabilities"]))
                require(delta <= 1e-4, "Combined vs fold archived probability mismatch")
                max_delta = max(max_delta, delta)
                row.update(prediction=median_label(p), entropy=-sum(x * math.log(x) for x in p if x > 0))
                eligible.append(row)
            seen.append(sid)
    require(Counter(seen) == Counter(expected.keys()), "Outer-fold partition is not a unique cover")
    require(len(eligible) == protocol["eligible_count"] and len(abstained) == protocol["mandatory_no_input_review_count"],
            "Eligibility differs from protocol")
    baseline = counts(eligible)
    archived_metrics = archived["metrics"]["fixed_median_0.5"]
    require(baseline["correct"] == archived_metrics["correct_count"]
            and baseline["severe_to_mild"] == archived_metrics["severe_to_mild_count"]
            and baseline["cross_two"] == archived_metrics["two_level_error_count"], "Recount differs from archive")
    audit = {"cohort_count": len(expected), "eligible_count": len(eligible), "abstained_count": len(abstained),
             "old_test_overlap": 0, "verified_classifiers": len(cv["folds"]),
             "archive_probability_max_difference": max_delta,
             "probability_check_scope": "combined archive versus per-fold archive; model forward pass not repeated",
             "baseline": baseline, "baseline_accuracy_all117": baseline["correct"] / len(expected),
             "assigned_class_support": [sum(label == i for label in expected.values()) for i in range(3)],
             "abstained_class_support": [sum(row["label"] == i for row in abstained) for i in range(3)],
             "eligible_modality_count_distribution": dict(sorted(Counter(r["modality_count"] for r in eligible).items()))}
    return eligible, abstained, audit, files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("csv", "cv-protocol", "archive", "protocol", "private-output", "summary"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    args = parser.parse_args()
    require(not args.summary.exists() and not args.private_output.exists(), "Never overwrite a prior attempt")
    protocol = load(args.protocol)
    require(bool(protocol.get("tie_salt")), "Require a fixed hash salt; sample IDs encode categories")
    rows, abstained, audit, files = audit_inputs(args, protocol)
    grouped = {fold: [r for r in rows if r["fold"] == fold] for fold in sorted({r["fold"] for r in rows})}
    fold_counts = {fold: len(items) for fold, items in grouped.items()}
    outcomes, selections = [], []
    for budget in protocol["budgets"]:
        quotas = allocate(fold_counts, budget)
        methods = {}
        def tie_key(row):
            return hashlib.sha256((protocol["tie_salt"] + "\0" + row["sample_id"]).encode("utf-8")).hexdigest()

        for method, key in (("entropy", lambda row: (-row["entropy"], tie_key(row))),
                            ("missing_first", lambda row: (row["modality_count"], tie_key(row)))):
            selected = {row["sample_id"] for fold, items in grouped.items()
                        for row in sorted(items, key=key)[:quotas[fold]]}
            methods[method] = describe_selection(rows, selected, len(abstained))
            selections.extend({"budget": budget, "method": method, "sample_id": sid} for sid in sorted(selected))
        rng = random.Random(protocol["random_comparator"]["seed"])
        random_counts = {metric: [] for metric in ("severe_to_mild", "cross_two", "errors")}
        for _ in range(protocol["random_comparator"]["repetitions"]):
            chosen = [r for fold, items in grouped.items() for r in rng.sample(items, quotas[fold])]
            metrics = counts(chosen)
            for metric in random_counts:
                random_counts[metric].append(metrics[metric])
        methods["random"] = {key: empirical(values) for key, values in random_counts.items()}
        outcomes.append({"budget": budget, "fold_quotas": quotas, "methods": methods})
    primary = next(item for item in outcomes if item["budget"] == protocol["primary_budget"])
    methods = primary["methods"]
    candidate, comparator = methods["entropy"]["captured"], methods["missing_first"]["captured"]
    gate = {"severe_gain_vs_missing": candidate["severe_to_mild"] - comparator["severe_to_mild"],
            "cross_two_gain_vs_missing": candidate["cross_two"] - comparator["cross_two"],
            "candidate_severe_above_random_median": candidate["severe_to_mild"] > methods["random"]["severe_to_mild"]["median"]}
    gate["passed"] = (gate["severe_gain_vs_missing"] >= protocol["gate"]["candidate_minus_missing_severe_at_least"]
                      and gate["cross_two_gain_vs_missing"] >= protocol["gate"]["candidate_minus_missing_cross_two_at_least"]
                      and gate["candidate_severe_above_random_median"])
    args.private_output.mkdir(parents=True)
    (args.private_output / "records.private.json").write_text(json.dumps(rows + abstained, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.private_output / "selections.private.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")
    (args.private_output / "input-receipt.json").write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"study_id": protocol["study_id"], "seed": args.seed, "status": "completed_retrospective_signal_analysis",
              "protocol_sha256": sha(args.protocol), "script_sha256": sha(Path(__file__)), "input_audit": audit,
              "budgets": outcomes, "primary_gate": gate, "new_model_training": False,
              "new_model_forward_pass": False, "old_test22_evaluated": False, "doctor_review_performed": False,
              "private_evidence": {"path": str(args.private_output),
                                   "records_sha256": sha(args.private_output / "records.private.json"),
                                   "selections_sha256": sha(args.private_output / "selections.private.json"),
                                   "input_receipt_sha256": sha(args.private_output / "input-receipt.json")}}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"seed": args.seed, "baseline_correct": audit["baseline"]["correct"],
                      "primary_gate": gate, "summary": str(args.summary)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
