"""Independently audit archived replay rows, weights and all training log lines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    parser.add_argument("--old-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = torch.load(args.old_checkpoint, map_location="cpu", weights_only=False)
    new = torch.load(args.output / "retraining/checkpoint.pt", map_location="cpu", weights_only=False)
    assert old["model"].keys() == new["model"].keys()
    different = [key for key in old["model"] if not torch.equal(old["model"][key], new["model"][key])]
    maximum = max((old["model"][key].float() - new["model"][key].float()).abs().max().item() for key in old["model"])
    pattern = r"epoch=\d+ train_loss=.*?(?=\n|$)"
    old_lines = re.findall(pattern, args.old_log.read_text(encoding="utf-8", errors="replace"))
    new_lines = re.findall(pattern, (args.output / "retraining-console.log").read_text(encoding="utf-8", errors="replace"))
    row_checks = {}
    for name in ("saved-model", "retrained-model"):
        rows = json.loads((args.output / f"{name}-predictions.private.json").read_text(encoding="utf-8"))
        matrix = [[0] * 3 for _ in range(3)]
        for row in rows:
            matrix[row["true_label"]][row["predicted_label"]] += 1
        result = dict(count=len(rows), correct=sum(matrix[i][i] for i in range(3)),
            absolute_error_sum=sum(abs(row["true_label"] - row["predicted_label"]) for row in rows),
            confusion_matrix=matrix)
        assert result["count"] == 21 and result["correct"] == 15 and result["absolute_error_sum"] == 8
        assert matrix == [[10, 0, 0], [4, 1, 0], [2, 0, 4]]
        row_checks[name] = result
    result = dict(parameter_tensor_count=len(old["model"]), equal_parameter_tensors=len(old["model"]) - len(different),
        max_parameter_absolute_difference=maximum, all_model_tensors_equal=not different,
        old_epoch_lines=len(old_lines), new_epoch_lines=len(new_lines),
        matching_logged_epoch_metrics=sum(a == b for a, b in zip(old_lines, new_lines)),
        all_logged_epoch_metrics_equal=old_lines == new_lines, independent_row_counts=row_checks)
    assert not different and maximum == 0 and old_lines == new_lines and len(old_lines) == 32
    (args.output / "independent-audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
