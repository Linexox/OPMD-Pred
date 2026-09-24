import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

from opmd_pred.data.zenodo import CATEGORIES, ZenodoImageDataset, split_by_patient
from opmd_pred.model.vit import ZenodoViTClassifier
from opmd_pred.scripts.common import configure_logger, device_name, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/zenodo/Imagewise_Data.csv")
    parser.add_argument("--images", default="data/zenodo/Images")
    parser.add_argument("--output", default="outputs/zenodo_vit")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--trainable-blocks", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-init", action="store_true")
    return parser.parse_args()


def run_epoch(model, loader, optimizer, device, class_weights, desc):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    progress = tqdm(loader, desc=desc, unit="batch")
    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = model(image)
        loss = torch.nn.functional.cross_entropy(logits, label, weight=class_weights)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * label.size(0)
        total_correct += (logits.argmax(1) == label).sum().item()
        total_count += label.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / total_count, total_correct / total_count


def main():
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output)
    configure_logger(output)
    device = torch.device(device_name())
    train_indices, validation_indices, test_indices = split_by_patient(args.csv, seed=args.seed)
    train_set = ZenodoImageDataset(args.csv, args.images, train_indices, train=True, cache_path=args.cache)
    validation_set = ZenodoImageDataset(
        args.csv, args.images, validation_indices, train=False, cache_path=args.cache
    )
    test_set = ZenodoImageDataset(args.csv, args.images, test_indices, train=False, cache_path=args.cache)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, **loader_kwargs)

    model = ZenodoViTClassifier(
        pretrained=not args.random_init,
        trainable_blocks=args.trainable_blocks,
    ).to(device)
    counts = torch.bincount(
        torch.tensor([CATEGORIES[row["Category"]] for row in train_set.rows]),
        minlength=len(CATEGORIES),
    ).float()
    class_weights = (counts.sum() / counts).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, optimizer, device, class_weights, f"epoch {epoch}/{args.epochs} train"
        )
        with torch.no_grad():
            validation_loss, validation_accuracy = run_epoch(
                model, validation_loader, None, device, class_weights, f"epoch {epoch}/{args.epochs} val"
            )
        logger.info(
            "epoch={} train_loss={:.4f} train_acc={:.4f} val_loss={:.4f} val_acc={:.4f}",
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
                    "image_encoder": model.image_encoder.state_dict(),
                    "class_count": 4,
                    "epoch": epoch,
                },
                output / "checkpoint.pt",
            )
    checkpoint = torch.load(output / "checkpoint.pt", map_location=device)
    model.image_encoder.load_state_dict(checkpoint["image_encoder"])
    with torch.no_grad():
        test_loss, test_accuracy = run_epoch(model, test_loader, None, device, class_weights, "test")
    logger.info("test_loss={:.4f} test_acc={:.4f}", test_loss, test_accuracy)


if __name__ == "__main__":
    main()
