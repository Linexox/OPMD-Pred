import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from opmd_pred.analysis.evaluation import evaluate_coalitions
from opmd_pred.analysis.shapley import interaction_values, shapley_values
from opmd_pred.data.sysu import SysuDataset, load_sysu_records, split_sysu_records
from opmd_pred.model.fusion import MultimodalOrdinalModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/SYSU_processed/sysu_samples.csv")
    parser.add_argument("--images", default="data/SYSU_processed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/explainability")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def coalition_name(coalition, names):
    selected = [name for index, name in enumerate(names) if coalition & (1 << index)]
    return "+".join(selected) if selected else "none"


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_sysu_records(args.csv)
    _, validation_records = split_sysu_records(records, seed=args.seed)
    validation_set = SysuDataset(validation_records, args.images, train=False)
    loader = DataLoader(validation_set, batch_size=args.batch_size, num_workers=0)
    model = MultimodalOrdinalModel(
        dimension=args.dimension,
        pretrained_image=False,
        tct_categories=max(record.tct_id for record in records),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    logits, labels, names, coalition_metrics = evaluate_coalitions(model, loader, device)
    values = logits.sigmoid().sum(dim=-1)
    shapley = shapley_values(values, len(names))
    interactions = interaction_values(values, len(names))
    full_mask = (1 << len(names)) - 1
    interaction_scale = values[:, full_mask].abs().mean().clamp_min(1e-8)
    interaction_ratio = interactions.abs().mean(dim=0) / interaction_scale

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
        "global_interaction_ratio": interaction_ratio.tolist(),
        "full_coalition": coalition_name(full_mask, names),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output / "samples.csv").open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["sample_index"]
        fieldnames.extend(f"shap_{name}" for name in names)
        fieldnames.extend(
            f"interaction_{names[first]}_{names[second]}"
            for first in range(len(names))
            for second in range(first + 1, len(names))
        )
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(shapley.size(0)):
            row = {"sample_index": row_index}
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
