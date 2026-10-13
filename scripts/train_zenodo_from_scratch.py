"""Train a ViT-B/16 on Zenodo from random initialization.

This is a temporary baseline script: it never loads ImageNet or another checkpoint.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from loguru import logger

from opmd_pred.data.zenodo import CATEGORIES, ZenodoImageDataset, split_by_patient
from opmd_pred.model.vit import ZenodoViTClassifier
from opmd_pred.scripts.common import configure_logger, device_name, set_seed


def run_epoch(model, loader, optimizer, device, class_weights, desc):
    training = optimizer is not None
    model.train(training)
    total_loss = total_weight = total_correct = total_count = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
            if training:
                loss.backward()
                optimizer.step()
            batch_weight = class_weights[labels].sum().item()
            total_loss += loss.item() * batch_weight
            total_weight += batch_weight
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    return total_loss / total_weight, total_correct / total_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Zenodo ViT-B/16 from random initialization."
    )
    parser.add_argument("--csv", default="data/zenodo/Imagewise_Data.csv")
    parser.add_argument("--images", default="data/zenodo/Images")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output", default="outputs/zenodo_vit_from_scratch")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed", type=int, default=42,
        help="Patient split seed, independent of the model initialization seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    configure_logger(output)
    (output / "args.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    device = torch.device(device_name())
    train_indices, validation_indices, test_indices = split_by_patient(
        args.csv, seed=args.split_seed
    )
    common = dict(
        csv_path=args.csv,
        image_root=args.images,
        cache_path=args.cache,
    )
    train_set = ZenodoImageDataset(
        **common, indices=train_indices, train=True
    )
    validation_set = ZenodoImageDataset(
        **common, indices=validation_indices, train=False
    )
    test_set = ZenodoImageDataset(
        **common, indices=test_indices, train=False
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, **loader_kwargs
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, **loader_kwargs
    )

    # Explicitly random initialization and full-model training.
    model = ZenodoViTClassifier(
        pretrained=False,
        class_count=len(CATEGORIES),
    ).to(device)
    # The shared classifier freezes blocks by default; undo that for scratch training.
    for parameter in model.parameters():
        parameter.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "from_scratch=True pretrained=False trainable_parameters={} total_parameters={}",
        trainable,
        total,
    )

    counts = torch.bincount(
        torch.tensor([CATEGORIES[row["Category"]] for row in train_set.rows]),
        minlength=len(CATEGORIES),
    ).float()
    class_weights = (counts.sum() / counts).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights,
            f"epoch {epoch}/{args.epochs} train",
        )
        with torch.no_grad():
            validation_loss, validation_accuracy = run_epoch(
                model,
                validation_loader,
                None,
                device,
                class_weights,
                f"epoch {epoch}/{args.epochs} val",
            )
        logger.info(
            "epoch={} train_loss={:.4f} train_acc={:.4f} "
            "val_loss={:.4f} val_acc={:.4f}",
            epoch,
            train_loss,
            train_accuracy,
            validation_loss,
            validation_accuracy,
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "model": model.state_dict(),
                    "image_encoder": model.image_encoder.state_dict(),
                    "class_count": len(CATEGORIES),
                    "epoch": epoch,
                    "from_scratch": True,
                    "args": vars(args),
                    "categories": CATEGORIES,
                    "validation_accuracy": validation_accuracy,
                },
                output / "checkpoint.pt",
            )

    checkpoint = torch.load(output / "checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        test_loss, test_accuracy = run_epoch(
            model, test_loader, None, device, class_weights, "test"
        )
    logger.info("test_loss={:.4f} test_acc={:.4f}", test_loss, test_accuracy)
    (output / "metrics.json").write_text(
        json.dumps({
            "best_epoch": checkpoint["epoch"],
            "best_validation_accuracy": best_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "from_scratch": True,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
