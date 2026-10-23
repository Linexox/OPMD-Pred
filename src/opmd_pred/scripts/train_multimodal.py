import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from loguru import logger

from opmd_pred.data.sysu import MODALITIES, SysuDataset, assign_tct_ids, build_tct_vocab, filter_observed_records, load_sysu_records, split_sysu_records
from opmd_pred.analysis.evaluation import classification_metrics as metrics
from opmd_pred.model.fusion import MultimodalOrdinalModel
from opmd_pred.model.losses import ordinal_loss
from opmd_pred.model.ordinal import ordinal_prediction
from opmd_pred.scripts.common import configure_logger, device_name, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/sysu/legacy/sysu_samples.csv")
    parser.add_argument("--images", default="data/processed/sysu/legacy")
    parser.add_argument("--image-checkpoint", default="")
    parser.add_argument("--fusion-checkpoint", default="")
    parser.add_argument("--output", default="outputs/experiments/sysu/sysu_multimodal")
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


def apply_modality_dropout(batch, probability, mode="bernoulli", names=MODALITIES):
    if probability == 0:
        return batch
    present = torch.stack(
        [batch[f"{name}_present"].bool() if name in {"image", "tct"}
         else batch[f"{name}_mask"].any(1) for name in names],
        dim=1,
    )
    if mode == "subset":
        dropped = torch.zeros_like(present)
        for row in range(present.size(0)):
            available = present[row].nonzero().flatten()
            if not available.numel() or torch.rand(()).item() >= probability:
                continue
            dropped[row] = present[row]
            keep_count = torch.randint(1, available.numel() + 1, (1,)).item()
            keep = available[torch.randperm(available.numel())[:keep_count]]
            dropped[row, keep] = False
    else:
        dropped = present & (torch.rand_like(present.float()) < probability)
    remaining = present & ~dropped
    for row in range(present.size(0)):
        if not remaining[row].any() and present[row].any():
            available = present[row].nonzero().flatten()
            retained = available[torch.randint(available.numel(), (1,)).item()]
            dropped[row, retained] = False
    batch = dict(batch)
    for index, name in enumerate(names):
        if name in {"image", "tct"}:
            key = f"{name}_present"
            batch[key] = batch[key].bool() & ~dropped[:, index]
        else:
            key = f"{name}_mask"
            batch[key] = batch[key].bool() & ~dropped[:, index].unsqueeze(1)
    return batch


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
        predictions = ordinal_prediction(logits.detach())
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
    logger.warning("Legacy exploratory sample split only; use train_sysu with reviewed patient manifest for clinical evaluation")
    device = torch.device(device_name())
    records = load_sysu_records(args.csv)
    records, excluded = filter_observed_records(records)
    train_records, validation_records = split_sysu_records(records, seed=args.seed)
    tct_vocab = build_tct_vocab(train_records)
    assign_tct_ids(records, tct_vocab)
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
        tct_categories=max(tct_vocab.values(), default=1),
    ).to(device)
    if args.image_checkpoint:
        checkpoint = torch.load(args.image_checkpoint, map_location=device)
        model.fusion.image_encoder.load_state_dict(checkpoint["image_encoder"])
    if args.fusion_checkpoint:
        checkpoint = torch.load(args.fusion_checkpoint, map_location=device)
        if checkpoint.get("tct_vocab") != tct_vocab:
            raise ValueError("Fusion checkpoint TCT vocabulary differs from the training-only vocabulary")
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
            "train_bal_acc={:.4f} train_mae={:.4f} train_qwk={} "
            "val_loss={:.4f} val_acc={:.4f} val_macro_f1={:.4f} "
            "val_bal_acc={:.4f} val_mae={:.4f} val_qwk={} val_cm={}",
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
                    "tct_vocab": tct_vocab,
                    "protocol": "legacy_exploratory_sample_split",
                    "excluded": excluded,
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
