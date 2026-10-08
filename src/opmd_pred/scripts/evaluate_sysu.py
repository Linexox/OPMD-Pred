"""Evaluate a fixed foundation checkpoint once; no test-time tuning or fitting."""

import argparse
from pathlib import Path

import torch

from opmd_pred.data.sysu import assign_tct_ids
from opmd_pred.data.sysu_protocol import file_sha256, select_records, split_coverage, validate_manifest
from opmd_pred.scripts.train_sysu import add_coverage_metrics, make_loader, read_checkpoint, restore_checkpoint, run_epoch, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing fixed-checkpoint evaluation")
    device = torch.device(args.device)
    checkpoint = read_checkpoint(args.checkpoint)
    records = validate_manifest(checkpoint["split_manifest"], args.csv, args.images)
    config = checkpoint["model_config"]
    splits, excluded = select_records(records, checkpoint["split_manifest"], config["modalities"])
    ids = [record.sample_id for record in splits[args.split]]
    if ids != checkpoint["effective_split_ids"][args.split] or excluded != checkpoint["excluded"]:
        raise ValueError("Effective split or modality exclusions differ from checkpoint")
    if not ids:
        raise ValueError("No eligible observed patients in requested split")
    assign_tct_ids(records, checkpoint["tct_vocab"])
    model, _ = restore_checkpoint(checkpoint, device)
    loader = make_loader(splits[args.split], args.images, config, args.batch_size, args.num_workers, device)
    metrics, labels, probabilities = run_epoch(model, loader, device, cost_matrix=checkpoint["cost_matrix"],
                                             ordinal_distance_weight=checkpoint["training_config"].get("ordinal_distance_weight", 0.0))
    coverage = split_coverage(checkpoint["split_manifest"], splits)
    add_coverage_metrics(metrics, coverage[args.split])
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"checkpoint_sha256": file_sha256(args.checkpoint), "epoch": checkpoint["epoch"],
                        "split_manifest_sha256": checkpoint["split_manifest_sha256"], "split": args.split,
                        "sample_ids": ids, "labels": labels.tolist(), "probabilities": probabilities.tolist(),
                        "metrics": metrics, "cost_matrix": checkpoint["cost_matrix"], "excluded": excluded})
    print(f"Fixed-checkpoint {args.split} results written to {output}")


if __name__ == "__main__":
    main()
