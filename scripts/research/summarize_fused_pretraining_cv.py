"""Recount audited fused-pretraining predictions and evaluate the frozen continuation rule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


METHODS = ("ordinal", "ordinal_pretrain", "fused_full_instance", "fused_full_class", "fused_subset_class")
LABELS = ("不做多模态预训练", "原单项特征配对", "融合全部资料，同病例作正例", "融合全部资料，同类别作正例", "全部与部分资料，同类别作正例")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = read(args.root / "independent-audit.json")
    assert audit["passed"] and audit["new_variants"] == 9 and audit["reference_variants"] == 6
    runs = audit["checks"]
    summary = {}
    for method, label in zip(METHODS, LABELS, strict=True):
        items = sorted([item for item in runs if item["method"] == method], key=lambda item: item["seed"])
        assert [item["seed"] for item in items] == [42, 43, 44]
        result = []
        for item in items:
            m = item["metrics"]
            cm = m["confusion_matrix"]
            assert m["assigned_count"] == 117 and m["eligible_count"] == 112 and m["abstained_count"] == 5
            assert [sum(row) for row in cm] == [52, 29, 31]
            result.append(dict(seed=item["seed"], **m,
                class_recall_all_assigned=[cm[y][y] / n for y, n in enumerate((55, 31, 31))],
                severe_to_mild_rate=m["severe_to_mild_count"] / 31,
                two_level_error_rate=m["two_level_error_count"] / 112))
        keys = ("accuracy_all_assigned", "correct_count", "severe_to_mild_count", "severe_to_mild_rate",
                "two_level_error_count", "two_level_error_rate", "mae", "abstained_count")
        means = {key: statistics.mean(item[key] for item in result) for key in keys}
        means["class_recall_all_assigned"] = [statistics.mean(item["class_recall_all_assigned"][y] for item in result) for y in range(3)]
        means["severe_recall_all_assigned"] = means["class_recall_all_assigned"][2]
        summary[method] = dict(label=label, per_seed=result, means=means)
    reference = summary["ordinal_pretrain"]["means"]
    gates = {}
    for method in METHODS[2:]:
        m = summary[method]["means"]
        checks = dict(accuracy_drop_within_two_percentage_points=m["accuracy_all_assigned"] >= reference["accuracy_all_assigned"] - .02 - 1e-12,
            severe_to_mild_average_reduced_at_least_one=m["severe_to_mild_count"] <= reference["severe_to_mild_count"] - 1 + 1e-12,
            two_level_errors_not_increased=m["two_level_error_count"] <= reference["two_level_error_count"] + 1e-12,
            moderate_recall_not_reduced=m["class_recall_all_assigned"][1] >= reference["class_recall_all_assigned"][1] - 1e-12)
        gates[method] = dict(passed=all(checks.values()), conditions=checks,
            accuracy_change_percentage_points=100 * (m["accuracy_all_assigned"] - reference["accuracy_all_assigned"]),
            severe_to_mild_average_change=m["severe_to_mild_count"] - reference["severe_to_mild_count"],
            two_level_errors_average_change=m["two_level_error_count"] - reference["two_level_error_count"])
    contrasts = {}
    for before, after in (("fused_full_instance", "fused_full_class"), ("fused_full_class", "fused_subset_class")):
        a, b = summary[before]["means"], summary[after]["means"]
        contrasts[after + "_versus_" + before] = dict(accuracy_change_percentage_points=100 * (b["accuracy_all_assigned"] - a["accuracy_all_assigned"]),
            severe_to_mild_average_change=b["severe_to_mild_count"] - a["severe_to_mild_count"],
            moderate_recall_change_percentage_points=100 * (b["class_recall_all_assigned"][1] - a["class_recall_all_assigned"][1]))
    public = dict(evaluation_scope=dict(records=117, labels=[55, 31, 31], predicted=112, abstained=5,
                  random_seed_bases=[42, 43, 44], folds=5, no_locked_test_evaluation=True, no_app_model_change=True),
        results=summary, continuation_rule_reference="ordinal_pretrain", continuation_checks=gates,
        paired_design_contrasts=contrasts, independent_audit={k: v for k, v in audit.items() if k != "checks"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(results={name: item["means"] for name, item in summary.items()},
                          continuation_checks=gates, contrasts=contrasts), ensure_ascii=False))


if __name__ == "__main__":
    main()
