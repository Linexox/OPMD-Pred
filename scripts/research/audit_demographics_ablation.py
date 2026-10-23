"""Independently recount the paired 117-record age/sex ablation predictions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predictions(rows):
    result = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in result:
            raise ValueError(f"Duplicate record: {sample_id}")
        probabilities = row["probabilities"]
        if row["status"] == "abstain":
            if probabilities is not None:
                raise ValueError(f"Abstention has probabilities: {sample_id}")
            predicted = None
        else:
            if probabilities is None or len(probabilities) != 3:
                raise ValueError(f"Missing class probabilities: {sample_id}")
            if any(not math.isfinite(value) or value < -1e-7 for value in probabilities):
                raise ValueError(f"Invalid probabilities: {sample_id}")
            if abs(sum(probabilities) - 1) > 1e-5:
                raise ValueError(f"Probabilities do not sum to 1: {sample_id}")
            # The project's fixed ordinal decision: count cumulative probabilities > 0.5.
            predicted = int(probabilities[1] + probabilities[2] > 0.5) + int(probabilities[2] > 0.5)
        result[sample_id] = {"label": int(row["label"]), "predicted": predicted}
    if len(result) != 117:
        raise ValueError(f"Expected 117 records, got {len(result)}")
    return result


def score(rows):
    support = [sum(row["label"] == label for row in rows.values()) for label in range(3)]
    correct = sum(row["predicted"] == row["label"] for row in rows.values())
    recall = [sum(row["label"] == label and row["predicted"] == label for row in rows.values()) / support[label]
              for label in range(3)]
    severe_to_mild = sum(row["label"] == 2 and row["predicted"] == 0 for row in rows.values())
    two_level = sum((row["label"], row["predicted"]) in {(0, 2), (2, 0)} for row in rows.values())
    assigned = [row for row in rows.values() if row["predicted"] is not None]
    mae = sum(abs(row["label"] - row["predicted"]) for row in assigned) / len(assigned)
    return {
        "accuracy": correct / len(rows),
        "correct_count": correct,
        "class_recall": recall,
        "severe_to_mild_count": severe_to_mild,
        "two_level_error_count": two_level,
        "mae_assigned_only": mae,
        "abstained_count": len(rows) - len(assigned),
        "class_support": support,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Directory containing ordinal_pretrain_seed42/43/44/oof_results.json")
    parser.add_argument("--demographics", type=Path, required=True,
                        help="Directory containing ordinal_pretrain_demographics_seed42/43/44/oof_results.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups = {}
    paired = {}
    for seed in (42, 43, 44):
        base_path = args.baseline / f"ordinal_pretrain_seed{seed}" / "oof_results.json"
        demo_path = args.demographics / f"ordinal_pretrain_demographics_seed{seed}" / "oof_results.json"
        base_result, demo_result = load(base_path), load(demo_path)
        base_rows = predictions(base_result["records"])
        demo_rows = predictions(demo_result["records"])
        if base_rows.keys() != demo_rows.keys():
            raise ValueError(f"The compared models cover different records for seed {seed}")
        if any(base_rows[sid]["label"] != demo_rows[sid]["label"] for sid in base_rows):
            raise ValueError(f"Ground truth differs for seed {seed}")
        base_score, demo_score = score(base_rows), score(demo_rows)
        for method, actual, stored in (("baseline", base_score, base_result["metrics"]),
                                       ("demographics", demo_score, demo_result["metrics"])):
            if actual["correct_count"] != stored["correct_count"]:
                raise ValueError(f"Recount differs from saved {method} accuracy for seed {seed}")
            if actual["severe_to_mild_count"] != stored["severe_to_mild_count"]:
                raise ValueError(f"Recount differs from saved {method} severe errors for seed {seed}")
        groups.setdefault("without_age_sex", []).append({"seed": seed, **base_score})
        groups.setdefault("with_age_sex", []).append({"seed": seed, **demo_score})
        paired[str(seed)] = {
            "baseline_accuracy": base_score["accuracy"],
            "age_sex_accuracy": demo_score["accuracy"],
            "accuracy_difference_pp": 100 * (demo_score["accuracy"] - base_score["accuracy"]),
            "baseline_class_recall": base_score["class_recall"],
            "age_sex_class_recall": demo_score["class_recall"],
            "baseline_severe_to_mild": base_score["severe_to_mild_count"],
            "age_sex_severe_to_mild": demo_score["severe_to_mild_count"],
        }

    summary = {name: {
        "per_seed": rows,
        "mean_accuracy": sum(row["accuracy"] for row in rows) / len(rows),
        "mean_class_recall": [sum(row["class_recall"][i] for row in rows) / len(rows) for i in range(3)],
        "mean_severe_to_mild_count": sum(row["severe_to_mild_count"] for row in rows) / len(rows),
        "mean_two_level_error_count": sum(row["two_level_error_count"] for row in rows) / len(rows),
        "mean_mae_assigned_only": sum(row["mae_assigned_only"] for row in rows) / len(rows),
    } for name, rows in groups.items()}
    output = {"passed": True, "independent_recount_from_saved_probabilities": True,
              "cohort_records": 117, "locked_test_used": False,
              "paired_seed_results": paired, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
