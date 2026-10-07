import argparse
import csv
import json
from pathlib import Path

import torch

from opmd_pred.analysis.evaluation import evaluate_coalitions
from opmd_pred.analysis.shapley import intershap_summary, interaction_values, shapley_values
from opmd_pred.data.sysu import assign_tct_ids
from opmd_pred.data.sysu_protocol import select_records, validate_manifest
from opmd_pred.model.ordinal import model_probabilities
from opmd_pred.scripts.train_sysu import make_loader, read_checkpoint, restore_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/SYSU_processed/sysu_samples.csv")
    parser.add_argument("--images", default="data/SYSU_processed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/explainability")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target", choices=["severe", "at_least_moderate", "expected_severity"], default="severe")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def coalition_name(coalition, names):
    selected = [name for index, name in enumerate(names) if coalition & (1 << index)]
    return "+".join(selected) if selected else "none"


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint = read_checkpoint(args.checkpoint)
    records = validate_manifest(checkpoint["split_manifest"], args.csv, args.images)
    splits, _ = select_records(records, checkpoint["split_manifest"], checkpoint["model_config"]["modalities"])
    validation_records = splits["val"]
    if [record.sample_id for record in validation_records] != checkpoint["effective_split_ids"]["val"]:
        raise ValueError("Validation IDs differ from saved protocol")
    assign_tct_ids(records, checkpoint["tct_vocab"])
    model, _ = restore_checkpoint(checkpoint, device)
    loader = make_loader(validation_records, args.images, checkpoint["model_config"], args.batch_size, 0, device)

    logits, labels, names, coalition_metrics = evaluate_coalitions(model, loader, device)
    probabilities = model_probabilities(logits, model.head_kind)
    if args.target == "severe":
        values = probabilities[..., 2]
    elif args.target == "at_least_moderate":
        values = probabilities[..., 1:].sum(dim=-1)
    else:
        values = probabilities[..., 1] + 2 * probabilities[..., 2]
    shapley = shapley_values(values, len(names))
    interactions = interaction_values(values, len(names))
    full_sii, intershap = intershap_summary(shapley, interactions)
    full_mask = (1 << len(names)) - 1

    metrics = {
        coalition_name(int(coalition), names): coalition_metrics[str(coalition)]
        for coalition in range(logits.size(1))
    }
    summary = {
        "checkpoint": str(args.checkpoint),
        "validation_count": len(validation_records),
        "modality_names": list(names),
        "coalition_metrics": metrics,
        "global_mean_signed_shap": shapley.mean(dim=0).tolist(),
        "global_mean_abs_shap": shapley.abs().mean(dim=0).tolist(),
        "global_mean_interaction": interactions.mean(dim=0).tolist(),
        "global_mean_abs_interaction": interactions.abs().mean(dim=0).tolist(),
        "global_mean_full_sii": full_sii.mean(dim=0).tolist(),
        "global_mean_abs_full_sii_descriptive": full_sii.abs().mean(dim=0).tolist(),
        "intershap": intershap,
        "target": args.target,
        "split": "val",
        "explanation_game": "observed_modalities_removed_via_mask; absent_modalities_remain_absent",
        "base_value_mean": values[:, 0].mean().item(),
        "note": "InterSHAP formulas apply to this scalar missing-token coalition game; no background integration or causal claim. Mean-absolute descriptive matrices differ from abs-mean global InterSHAP.",
        "full_coalition": coalition_name(full_mask, names),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    with (output / "samples.csv").open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["sample_index", "sample_id", "base_value", "model_value"]
        fieldnames.extend(f"shap_{name}" for name in names)
        fieldnames.extend(
            f"interaction_{names[first]}_{names[second]}"
            for first in range(len(names))
            for second in range(first + 1, len(names))
        )
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(shapley.size(0)):
            row = {"sample_index": row_index, "sample_id": validation_records[row_index].sample_id,
                   "base_value": values[row_index, 0].item(), "model_value": values[row_index, full_mask].item()}
            row.update(
                {f"shap_{name}": float(shapley[row_index, index]) for index, name in enumerate(names)}
            )
            row.update(
                {
                    f"interaction_{names[first]}_{names[second]}": float(
                        interactions[row_index, first, second]
                    )
                    for first in range(len(names))
                    for second in range(first + 1, len(names))
                }
            )
            writer.writerow(row)


if __name__ == "__main__":
    main()
