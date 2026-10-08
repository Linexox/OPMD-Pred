"""SYSU baseline: reviewed patient split, five modalities, train/val only."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from opmd_pred.analysis.evaluation import classification_metrics
from opmd_pred.data.sysu import MODALITIES, SysuDataset, assign_tct_ids, build_tct_vocab
from opmd_pred.data.sysu_protocol import PROTOCOL, file_sha256, load_manifest, manifest_digest, select_records, split_coverage
from opmd_pred.model.fusion import MultimodalOrdinalModel
from opmd_pred.model.losses import classification_loss, cumulative_distance_loss
from opmd_pred.model.ordinal import model_probabilities, probability_decision
from opmd_pred.scripts.common import set_seed
from opmd_pred.scripts.train_multimodal import apply_modality_dropout, move_batch


CHECKPOINT_FORMAT = "sysu-foundation-v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--protocol", choices=[PROTOCOL], default=PROTOCOL)
    parser.add_argument("--image-encoder", choices=["dinov2_b14", "vit_b16"], default="dinov2_b14")
    parser.add_argument("--image-checkpoint", default="", help="Required when image is selected; no pretrained downloads")
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--fusion", choices=["transformer", "late"], default="transformer")
    parser.add_argument("--head", choices=["softmax", "ordinal", "corn"], default="softmax")
    parser.add_argument("--ordinal-distance-weight", type=float, default=0.0,
                        help="Research-only squared CDF target loss coefficient; default disables it")
    parser.add_argument("--cost-matrix", default="", help="Opt-in JSON 3x3 matrix: rows=true, columns=decision; no clinical default")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--modality-dropout", type=float, default=0.15)
    parser.add_argument("--dropout-mode", choices=["bernoulli", "subset"], default="bernoulli")
    parser.add_argument("--unfreeze-image-blocks", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if len(set(args.modalities)) != len(args.modalities):
        parser.error("Select each modality once")
    if "image" in args.modalities and not args.image_checkpoint:
        parser.error("Image modality requires --image-checkpoint; random or downloaded encoder initialization is disabled")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.num_workers < 0 or args.lr <= 0:
        parser.error("Use positive epochs/batch-size/patience/lr and nonnegative num-workers")
    if args.dimension < 4 or args.dimension % 4 or not 0 <= args.modality_dropout <= 1:
        parser.error("Dimension must be divisible by 4; modality-dropout must be in [0,1]")
    if not 0 <= args.ordinal_distance_weight < float("inf"):
        parser.error("ordinal-distance-weight must be finite and nonnegative")
    return args


def read_cost_matrix(path):
    if not path:
        return None
    matrix = torch.tensor(json.loads(Path(path).read_text(encoding="utf-8")), dtype=torch.float32)
    if matrix.shape != (3, 3) or not torch.isfinite(matrix).all() or (matrix < 0).any() or matrix.diag().ne(0).any():
        raise ValueError("Cost matrix must be finite nonnegative 3x3 with zero diagonal; rows=true, columns=action")
    return matrix.tolist()


def model_from_config(config):
    return MultimodalOrdinalModel(
        dimension=config["dimension"], pretrained_image=False,
        tct_categories=config["tct_categories"], image_encoder=config["image_encoder"],
        modalities=config["modalities"], head=config["head"], fusion_mode=config["fusion"],
    )


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Use a new SYSU foundation checkpoint; legacy demographic checkpoints are incompatible")
    if manifest_digest(checkpoint["split_manifest"]) != checkpoint["split_manifest_sha256"]:
        raise ValueError("Saved manifest canonical digest does not match checkpoint")
    return checkpoint


def restore_checkpoint(checkpoint, device):
    model = model_from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.fusion.freeze_image()
    return model.to(device).eval(), checkpoint


def make_loader(records, images, config, batch_size, workers, device, training=False):
    dataset = SysuDataset(records, images, train=training,
                          image_encoder=config["image_encoder"], modalities=config["modalities"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=training,
                      num_workers=workers, pin_memory=device.type == "cuda")


def run_epoch(model, loader, device, optimizer=None, probability=0.0, dropout_mode="bernoulli", cost_matrix=None,
              ordinal_distance_weight=0.0):
    training = optimizer is not None
    model.train(training)
    labels, probabilities, total_loss, total_probability_loss, total_distance = [], [], 0.0, 0.0, 0.0
    probability_normalizer = 0
    for batch in loader:
        batch = move_batch(batch, device)
        if training:
            batch = apply_modality_dropout(batch, probability, dropout_mode, model.fusion.names)
        with torch.set_grad_enabled(training):
            logits = model(batch)
            probability_loss = classification_loss(logits, batch["label"], model.head_kind)
            distance = cumulative_distance_loss(model_probabilities(logits, model.head_kind), batch["label"])
            loss = probability_loss + ordinal_distance_weight * distance
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        labels.append(batch["label"].detach().cpu())
        probabilities.append(model_probabilities(logits.detach(), model.head_kind).cpu())
        total_loss += loss.item() * batch["label"].numel()
        count = batch["label"].numel()
        task_count = count + int((batch["label"] > 0).sum().item()) if model.head_kind == "corn" and not training else count
        total_probability_loss += probability_loss.item() * task_count
        probability_normalizer += task_count
        total_distance += distance.item() * batch["label"].numel()
    labels, probabilities = torch.cat(labels), torch.cat(probabilities)
    argmax = probability_decision(probabilities)
    summary = classification_metrics(argmax, labels)
    if cost_matrix is not None:
        costs = torch.tensor(cost_matrix, dtype=probabilities.dtype)
        decisions = probability_decision(probabilities, costs)
        cost_summary = classification_metrics(decisions, labels)
        cost_summary["mean_cost"] = costs[labels, decisions].mean().item()
        summary["cost_decision"] = cost_summary
        summary["argmax_mean_cost"] = costs[labels, argmax].mean().item()
    summary["probability_loss"] = total_probability_loss / probability_normalizer
    summary["probability_loss_aggregation"] = (
        "global_conditional_task_mean" if model.head_kind == "corn" and not training else "record_weighted_minibatch_mean"
    )
    summary["squared_cumulative_distance"] = total_distance / labels.numel()
    summary["loss"] = (
        summary["probability_loss"] + ordinal_distance_weight * summary["squared_cumulative_distance"]
        if model.head_kind == "corn" and not training else total_loss / labels.numel()
    )
    summary["ordinal_distance_weight"] = ordinal_distance_weight
    return summary, labels, probabilities


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def add_coverage_metrics(metrics, coverage):
    metrics["coverage"] = coverage
    correct_count = sum(metrics["confusion_matrix"][label][label] for label in range(3))
    metrics["accuracy_all_assigned"] = correct_count / coverage["assigned_count"]
    severe_support = coverage["assigned_class_support"][2]
    metrics["severe_recall_all_assigned"] = metrics["confusion_matrix"][2][2] / severe_support if severe_support else None
    metrics["abstention_policy"] = "no_observed_selected_input; counted_as_not_correct_in_all_assigned_accuracy"
    if "cost_decision" in metrics:
        cost_correct = sum(metrics["cost_decision"]["confusion_matrix"][label][label] for label in range(3))
        metrics["cost_decision"]["accuracy_all_assigned"] = cost_correct / coverage["assigned_count"]
        metrics["cost_decision"]["severe_recall_all_assigned"] = (
            metrics["cost_decision"]["confusion_matrix"][2][2] / severe_support if severe_support else None
        )


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a new output directory; existing experiment artifacts will not be overwritten")
    # Resolve data integrity before model construction or choosing a compute device.
    manifest, records = load_manifest(args.split_manifest, args.csv, args.images)
    splits, excluded = select_records(records, manifest, args.modalities)
    if not splits["train"] or not splits["val"]:
        raise ValueError("Selected modalities have no observed records in train or val")
    if {record.label for record in splits["train"]} != {0, 1, 2}:
        raise ValueError("Training eligible records must include all three classes; do not search seeds using the held-out test")
    coverage = split_coverage(manifest, splits)
    tct_vocab = build_tct_vocab(splits["train"])
    assign_tct_ids(records, tct_vocab)
    cost_matrix = read_cost_matrix(args.cost_matrix)
    config = {"dimension": args.dimension, "tct_categories": max(tct_vocab.values(), default=1),
              "image_encoder": args.image_encoder, "modalities": args.modalities,
              "fusion": args.fusion, "head": args.head}
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    set_seed(args.seed)
    model = model_from_config(config)
    encoder_source = None
    if "image" in args.modalities:
        artifact = model.fusion.image_encoder.load_artifact(args.image_checkpoint)
        encoder_source = {"path": str(Path(args.image_checkpoint).resolve()), "sha256": file_sha256(args.image_checkpoint),
                          "source_checkpoint_sha256": artifact.get("source_checkpoint_sha256"), "best_epoch": artifact.get("best_epoch"),
                          "public_weights_revision": artifact.get("public_weights_revision"),
                          "public_weights_sha256": artifact.get("public_weights_sha256"),
                          "feature_definition": artifact.get("feature_definition"),
                          "preprocessing": artifact.get("preprocessing"), "normalization": artifact.get("normalization")}
        del artifact
        model.fusion.unfreeze_image_blocks(args.unfreeze_image_blocks)
    model = model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(splits["train"], args.images, config, args.batch_size, args.num_workers, device, True)
    val_loader = make_loader(splits["val"], args.images, config, args.batch_size, args.num_workers, device)
    output.mkdir(parents=True)
    manifest_sha = manifest_digest(manifest)
    manifest_source_sha = file_sha256(args.split_manifest)
    effective_ids = {split: [record.sample_id for record in selected] for split, selected in splits.items()}
    write_json(output / "split_manifest.json", manifest)
    write_json(output / "config.json", {**vars(args), "model_config": config, "tct_vocab": tct_vocab,
                                        "tct_special_ids": {"missing": 0, "unknown_observed": 1},
                                        "cost_matrix": cost_matrix, "effective_split_ids": effective_ids, "excluded": excluded,
                                        "encoder_source": encoder_source, "split_manifest_sha256": manifest_sha,
                                        "split_manifest_source_sha256": manifest_source_sha, "split_coverage": coverage})
    history, best_key, stale, best_epoch = [], (-1.0, -1.0), 0, None
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, device, optimizer, args.modality_dropout, args.dropout_mode, cost_matrix,
                                      args.ordinal_distance_weight)
        val_metrics, _, _ = run_epoch(model, val_loader, device, cost_matrix=cost_matrix,
                                    ordinal_distance_weight=args.ordinal_distance_weight)
        add_coverage_metrics(train_metrics, coverage["train"])
        add_coverage_metrics(val_metrics, coverage["val"])
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        key = (val_metrics["accuracy_all_assigned"], val_metrics["macro_f1"])
        if key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            torch.save({"format": CHECKPOINT_FORMAT, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "validation_metrics": val_metrics, "selection_rule": "val_argmax_all_assigned_accuracy_then_eligible_macro_f1",
                        "model_config": config, "training_config": vars(args), "cost_matrix": cost_matrix,
                        "tct_vocab": tct_vocab, "tct_special_ids": {"missing": 0, "unknown_observed": 1},
                        "split_manifest": manifest, "split_manifest_sha256": manifest_sha,
                        "split_manifest_source_sha256": manifest_source_sha, "split_coverage": coverage,
                        "effective_split_ids": effective_ids, "excluded": excluded, "encoder_source": encoder_source,
                        "versions": {name: version(name) for name in ("torch", "torchvision", "timm")}, "history": list(history)}, output / "checkpoint.pt")
        else:
            stale += 1
        write_json(output / "history.json", {"history": history, "best_epoch": best_epoch, "test_evaluated": False})
        print(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics, "best_epoch": best_epoch}, ensure_ascii=False, allow_nan=False), flush=True)
        if stale >= args.patience:
            break
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    checkpoint["history"] = history
    torch.save(checkpoint, output / "checkpoint.pt")


if __name__ == "__main__":
    main()
