import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from loguru import logger

from opmd_pred.data.sysu import SysuDataset, load_sysu_records, split_sysu_records
from opmd_pred.model.fusion import MultimodalOrdinalModel
from opmd_pred.model.losses import ordinal_loss
from opmd_pred.scripts.common import configure_logger, device_name, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/SYSU_processed/sysu_samples.csv")
    parser.add_argument("--images", default="data/SYSU_processed")
    parser.add_argument("--image-checkpoint", default="")
    parser.add_argument("--fusion-checkpoint", default="")
    parser.add_argument("--output", default="outputs/sysu_multimodal")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--modality-dropout", type=float, default=0.15)
    parser.add_argument("--dropout-mode", choices=["bernoulli", "subset"], default="bernoulli")
    parser.add_argument("--unfreeze-image-blocks", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def apply_modality_dropout(batch, probability, mode="bernoulli"):
    if probability == 0:
        return batch
    names = ["image", "demo", "history", "tct", "dna", "methylation"]
    present = torch.stack(
        [
            batch["image_present"].bool(),
            batch["demo_mask"].any(1),
            batch["history_mask"].any(1),
            batch["tct_present"].bool(),
            batch["dna_mask"].any(1),
            batch["methylation_mask"].any(1),
        ],
        dim=1,
    )
    if mode == "subset":
        dropped = present.clone()
        for row in range(present.size(0)):
            available = present[row].nonzero().flatten()
            keep_count = torch.randint(1, available.numel() + 1, (1,)).item()
            keep = available[torch.randperm(available.numel())[:keep_count]]
            dropped[row, keep] = False
    else:
        dropped = present & (torch.rand_like(present.float()) < probability)
    remaining = present & ~dropped
    for row in range(present.size(0)):
        if not remaining[row].any() and present[row].any():
            dropped[row, present[row].nonzero()[0].item()] = False
    batch = dict(batch)
    batch["image_present"] = batch["image_present"] & ~dropped[:, 0]
    batch["demo_mask"] = batch["demo_mask"] * (~dropped[:, 1]).unsqueeze(1)
    batch["history_mask"] = batch["history_mask"] * (~dropped[:, 2]).unsqueeze(1)
    batch["tct_present"] = batch["tct_present"] & ~dropped[:, 3]
    batch["dna_mask"] = batch["dna_mask"] & ~dropped[:, 4].unsqueeze(1)
    batch["methylation_mask"] = batch["methylation_mask"] & ~dropped[:, 5].unsqueeze(1)
    return batch


def metrics(predictions, labels, class_count=3):
    confusion = torch.bincount(
        labels * class_count + predictions,
        minlength=class_count * class_count,
    ).reshape(class_count, class_count).float()
    support = confusion.sum(dim=1)
    true_positive = confusion.diag()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    macro_f1 = f1.mean().item()
    balanced_accuracy = recall[support > 0].mean().item()

    total = confusion.sum()
    observed = torch.arange(class_count, device=confusion.device, dtype=torch.float32)
    weights = (observed[:, None] - observed[None, :]).square()
    observed_disagreement = (weights * confusion / total).sum()
    expected = support[:, None] * confusion.sum(dim=0)[None, :] / total
    expected_disagreement = (weights * expected / total).sum()
    qwk = 1.0 - (observed_disagreement / expected_disagreement).item()
    return {
        "mae": (predictions - labels).abs().float().mean().item(),
        "accuracy": (predictions == labels).float().mean().item(),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "qwk": qwk,
        "confusion_matrix": confusion.to(torch.long).tolist(),
    }


def run_epoch(model, loader, optimizer, device, dropout_probability=0.0, dropout_mode="bernoulli"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_mae = 0.0
    total_correct = 0
    total_count = 0
    all_predictions = []
    all_labels = []
    for batch in loader:
        batch = move_batch(batch, device)
        if training:
            batch = apply_modality_dropout(batch, dropout_probability, dropout_mode)
        logits = model(batch)
        loss = ordinal_loss(logits, batch["label"])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        predictions = (logits.detach().sigmoid() > 0.5).sum(dim=1)
        current = metrics(predictions, batch["label"])
        all_predictions.append(predictions.cpu())
        all_labels.append(batch["label"].cpu())
        count = batch["label"].size(0)
        total_loss += loss.item() * count
        total_mae += current["mae"] * count
        total_correct += current["accuracy"] * count
        total_count += count
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    summary = metrics(predictions, labels)
    summary["mae"] = total_mae / total_count
    summary["accuracy"] = total_correct / total_count
    return total_loss / total_count, summary


def main():
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output)
    configure_logger(output)
    device = torch.device(device_name())
    records = load_sysu_records(args.csv)
    train_records, validation_records = split_sysu_records(records, seed=args.seed)
    train_set = SysuDataset(train_records, args.images, train=True)
    validation_set = SysuDataset(validation_records, args.images, train=False)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_set, **loader_options)
    model = MultimodalOrdinalModel(
        dimension=args.dimension,
        pretrained_image=not bool(args.image_checkpoint),
        tct_categories=max(record.tct_id for record in records),
    ).to(device)
    if args.image_checkpoint:
        checkpoint = torch.load(args.image_checkpoint, map_location=device)
        model.fusion.image_encoder.load_state_dict(checkpoint["image_encoder"])
    if args.fusion_checkpoint:
        checkpoint = torch.load(args.fusion_checkpoint, map_location=device)
        model.fusion.load_state_dict(checkpoint["fusion"])
    if args.unfreeze_image_blocks:
        model.fusion.unfreeze_image_blocks(args.unfreeze_image_blocks)
    else:
        model.fusion.freeze_image()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    best_accuracy = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, optimizer, device, args.modality_dropout, args.dropout_mode
        )
        with torch.no_grad():
            validation_loss, validation_metrics = run_epoch(
                model, validation_loader, None, device
            )
        logger.info(
            "epoch={} train_loss={:.4f} train_acc={:.4f} train_macro_f1={:.4f} "
            "train_bal_acc={:.4f} train_mae={:.4f} train_qwk={:.4f} "
            "val_loss={:.4f} val_acc={:.4f} val_macro_f1={:.4f} "
            "val_bal_acc={:.4f} val_mae={:.4f} val_qwk={:.4f} val_cm={}",
            epoch,
            train_loss,
            train_metrics["accuracy"],
            train_metrics["macro_f1"],
            train_metrics["balanced_accuracy"],
            train_metrics["mae"],
            train_metrics["qwk"],
            validation_loss,
            validation_metrics["accuracy"],
            validation_metrics["macro_f1"],
            validation_metrics["balanced_accuracy"],
            validation_metrics["mae"],
            validation_metrics["qwk"],
            validation_metrics["confusion_matrix"],
        )
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_accuracy": best_accuracy,
                    "config": vars(args),
                    "validation_sample_ids": [record.sample_id for record in validation_records],
                    "modality_names": list(model.fusion.names),
                },
                output / "checkpoint.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                logger.info("early_stop epoch={} best_val_acc={:.4f}", epoch, best_accuracy)
                break


if __name__ == "__main__":
    main()
