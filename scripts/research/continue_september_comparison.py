"""Continue only prespecified qualifying candidates on initialization seeds 43 and 44."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time

import torch

SPEC = importlib.util.spec_from_file_location("september_study", Path(__file__).with_name("run_september_comparison.py"))
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


def qualifies(candidate, control):
    severe = (candidate["correct_count"] >= control["correct_count"] - 2
              and candidate["severe_to_mild_count"] <= control["severe_to_mild_count"] - 1
              and candidate["two_level_error_count"] <= control["two_level_error_count"])
    accuracy = (candidate["correct_count"] > control["correct_count"]
                and candidate["severe_to_mild_count"] <= control["severe_to_mild_count"]
                and candidate["two_level_error_count"] <= control["two_level_error_count"])
    return severe or accuracy


def args_from_receipt(path):
    receipt = json.loads(path.read_text())
    command = receipt["command"]
    values = {command[i][2:].replace("-", "_"): command[i + 1]
              for i in range(3, len(command), 2) if command[i].startswith("--")}
    return argparse.Namespace(**{k: Path(v) for k, v in values.items() if k not in ("phase", "seconds")}), receipt


def verify_control(args, protocol, result, seed):
    work = args.reference.parent
    archive_root = work / "sysu_raw_medical_multiseed_20261009" / f"seed{seed}"
    expected = {}
    for fold in protocol["folds"]:
        saved = json.loads((archive_root / f"fold{fold['fold']}/baseline/outer_predictions.json").read_text())
        for row in saved["records"]:
            expected[row["sample_id"]] = row
    max_error = 0.
    for row in result["records"]:
        previous = expected[row["sample_id"]]
        STUDY.require(previous["status"] == row["status"] and previous["label"] == row["label"], "New control coverage differs")
        if row["probabilities"] is not None:
            max_error = max(max_error, max(abs(a - b) for a, b in zip(previous["probabilities"], row["probabilities"])))
    STUDY.require(max_error <= 1e-4, "New control training does not reproduce archived run")
    STUDY.require(result["metrics"]["correct_count"] == {43: 63, 44: 65}[seed], "Control correct count differs")
    STUDY.write(args.output / f"control-seed{seed}-reproduction.json", dict(passed=True, max_absolute_probability_error=max_error,
                 full_training_repeated=True, old_test_new_inference=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=7200)
    cli = parser.parse_args()
    STUDY.DEADLINE = time.monotonic() + cli.seconds
    args, receipt = args_from_receipt(cli.receipt)
    torch.set_num_threads(2)
    STUDY.check_gpu()
    protocol = json.loads((args.input / "development_nested_cv_v3.json").read_text())
    core = json.loads((args.output / "summary-core.json").read_text())
    details = json.loads((args.output / "summary-details.json").read_text())
    control = core["dino_softmax_balanced"]
    selected = [name for name, m in dict(core, **details).items()
                if name != "dino_softmax_balanced" and qualifies(m, control)]
    selection = dict(selected=selected, source_sha256=STUDY.digest(__file__),
                     rule="severe-errors>=1 fewer, cross-two not higher, correct loss<=2; OR correct higher and both errors not higher",
                     evaluated_initialization=42, continuation_initializations=[43, 44],
                     selection_scope="exploratory choice after seed42; not an independent test")
    selection_path = args.output / "continuation-selection.json"
    if selection_path.exists():
        STUDY.require(json.loads(selection_path.read_text()) == selection, "Continuation choices changed")
    else:
        STUDY.write(selection_path, selection)
    results = {"dino_softmax_balanced": [control]}
    results.update({name: [dict(core, **details)[name]] for name in selected})
    for seed in (43, 44):
        config = STUDY.configuration(seed_offset=seed - 42)
        baseline = STUDY.train_variant(args, protocol, f"dino_softmax_balanced_seed{seed}", config)
        verify_control(args, protocol, baseline, seed)
        results["dino_softmax_balanced"].append(baseline["metrics"])
        for name in selected:
            original = json.loads((args.output / "runs" / name / "oof_results.json").read_text())
            config = dict(original["config"], seed_offset=seed - 42)
            result = STUDY.train_variant(args, protocol, f"{name}_seed{seed}", config)
            results[name].append(result["metrics"])
        STUDY.write(args.output / "continuation-progress.json", results)
    keys = ["correct_count", "accuracy_all_assigned", "severe_to_mild_count", "two_level_error_count", "severe_recall_all_assigned", "mae"]
    means = {name: {key: statistics.mean(m[key] for m in runs) for key in keys} for name, runs in results.items()}
    passed = {name: qualifies(m, means["dino_softmax_balanced"]) for name, m in means.items() if name != "dino_softmax_balanced"}
    STUDY.write(args.output / "multiseed-results.json", dict(per_initialization=results, means=means,
                continuation_conditions_satisfied_on_means=passed, independent_new_cases_evaluated=False))
    STUDY.write(args.output / "completed-continuation.json", dict(completed=True, selected=selected, old_test_new_inference=False))
    print(json.dumps(dict(event="continuation_complete", means=means, passed=passed)), flush=True)


if __name__ == "__main__":
    main()
