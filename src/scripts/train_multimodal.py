import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from loguru import logger

from src.data.sysu import SysuDataset, load_sysu_records, split_sysu_records
from src.model.fusion import MultimodalOrdinalModel
from src.model.losses import ordinal_loss
from src.scripts.common import configure_logger, device_name, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/SYSU_processed/sysu_samples.csv")
    parser.add_argument("--images", default="data/SYSU_processed")
    parser.add_argument("--image-checkpoint", default="")
    parser.add_argument("--fusion-checkpoint", default="")
    parser.add_argument("--output", default="outputs/sysu_multimodal")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--modality-dropout", type=float, default=0.15)
    parser.add_argument("--unfreeze-image-blocks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def apply_modality_dropout(batch, probability):
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


def metrics(logits, labels):
    predictions = (logits.sigmoid() > 0.5).sum(dim=1)
    return {
        "mae": (predictions - labels).abs().float().mean().item(),
        "accuracy": (predictions == labels).float().mean().item(),
    }


def run_epoch(model, loader, optimizer, device, dropout_probability=0.0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_mae = 0.0
    total_correct = 0
    total_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        if training:
            batch = apply_modality_dropout(batch, dropout_probability)
        logits = model(batch)
        loss = ordinal_loss(logits, batch["label"])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        current = metrics(logits.detach(), batch["label"])
        count = batch["label"].size(0)
        total_loss += loss.item() * count
        total_mae += current["mae"] * count
        total_correct += current["accuracy"] * count
        total_count += count
    return total_loss / total_count, total_mae / total_count, total_correct / total_count


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
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, num_workers=0)
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
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae, train_accuracy = run_epoch(
            model, train_loader, optimizer, device, args.modality_dropout
        )
        with torch.no_grad():
            validation_loss, validation_mae, validation_accuracy = run_epoch(
                model, validation_loader, None, device
            )
        logger.info(
            "epoch={} train_loss={:.4f} train_mae={:.4f} train_acc={:.4f} "
            "val_loss={:.4f} val_mae={:.4f} val_acc={:.4f}",
            epoch,
            train_loss,
            train_mae,
            train_accuracy,
            validation_loss,
            validation_mae,
            validation_accuracy,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "validation_loss": validation_loss},
                output / "checkpoint.pt",
            )


if __name__ == "__main__":
    main()
