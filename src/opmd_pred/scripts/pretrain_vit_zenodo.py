import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

from opmd_pred.data.zenodo import (
    CATEGORIES,
    ZenodoImageDataset,
    _load_rows,
    split_by_patient,
    split_stratified,
)
from opmd_pred.model.vit import ZenodoViTClassifier, interpolate_pos_embedding
from opmd_pred.scripts.common import configure_logger, device_name, multiclass_metrics, set_seed


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
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["patient", "image"], default="patient")
    parser.add_argument("--class-weight", choices=["inverse", "sqrt", "none"], default="inverse")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--select-by", choices=["val_acc", "val_loss"], default="val_acc")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--llrd-decay", type=float, default=1.0)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--pretrained-weights", choices=["v1", "swag_linear", "dinov2_vitb14"], default="v1")
    parser.add_argument("--mlp-head", action="store_true")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--exclude-images", default=None)
    return parser.parse_args()


def learning_rate_scale(args, index):
    if args.scheduler != "cosine":
        return 1.0
    warmup = max(args.warmup_epochs, 0)
    total = max(args.epochs, 1)
    if warmup and index < warmup:
        return (index + 1) / warmup
    progress = min(max((index - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def run_epoch(model, loader, optimizer, device, class_weights, desc, label_smoothing=0.0, mixup_alpha=0.0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    predictions = []
    labels = []
    progress = tqdm(loader, desc=desc, unit="batch")
    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        mixed_label = None
        if training and mixup_alpha > 0:
            lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
            index = torch.randperm(image.size(0), device=image.device)
            image = lam * image + (1.0 - lam) * image[index]
            mixed_label = label[index]
        logits = model(image)
        loss = torch.nn.functional.cross_entropy(
            logits, label, weight=class_weights, label_smoothing=label_smoothing
        )
        if mixed_label is not None:
            loss = lam * loss + (1.0 - lam) * torch.nn.functional.cross_entropy(
                logits, mixed_label, weight=class_weights, label_smoothing=label_smoothing
            )
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        prediction = logits.argmax(1)
        total_loss += loss.item() * label.size(0)
        total_correct += (prediction == label).sum().item()
        total_count += label.size(0)
        predictions.append(prediction.detach().cpu())
        labels.append(label.detach().cpu())
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return (
        total_loss / total_count,
        total_correct / total_count,
        torch.cat(predictions),
        torch.cat(labels),
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output)
    configure_logger(output)
    device = torch.device(device_name())
    if args.split_mode == "patient":
        train_indices, validation_indices, test_indices = split_by_patient(args.csv, seed=args.split_seed)
    else:
        train_indices, validation_indices, test_indices = split_stratified(args.csv, seed=args.split_seed)
    logger.info(
        "split_mode={} split_seed={} train={} validation={} test={}",
        args.split_mode,
        args.split_seed,
        len(train_indices),
        len(validation_indices),
        len(test_indices),
    )
    if args.exclude_images:
        import csv

        with open(args.exclude_images, encoding="utf-8") as file:
            excluded = {row["image_name"] for row in csv.DictReader(file)}
        all_rows = _load_rows(args.csv)
        before = len(train_indices)
        train_indices = [index for index in train_indices if all_rows[index]["Image Name"] not in excluded]
        logger.info(
            "exclude_images={} removed={} train {} -> {}",
            args.exclude_images, before - len(train_indices), before, len(train_indices),
        )
    train_set = ZenodoImageDataset(
        args.csv, args.images, train_indices, train=True, cache_path=args.cache, crop_size=args.img_size
    )
    validation_set = ZenodoImageDataset(
        args.csv, args.images, validation_indices, train=False, cache_path=args.cache, crop_size=args.img_size
    )
    test_set = ZenodoImageDataset(
        args.csv, args.images, test_indices, train=False, cache_path=args.cache, crop_size=args.img_size
    )
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
        weights_name=args.pretrained_weights,
        mlp_head=args.mlp_head,
        img_size=args.img_size,
    ).to(device)
    counts = torch.bincount(
        torch.tensor([CATEGORIES[row["Category"]] for row in train_set.rows]),
        minlength=len(CATEGORIES),
    ).float()
    if args.class_weight == "none":
        class_weights = None
    else:
        class_weights = counts.sum() / counts
        if args.class_weight == "sqrt":
            class_weights = class_weights.sqrt()
        class_weights = class_weights.to(device)
    logger.info(
        "class_weight={} weights={}",
        args.class_weight,
        None if class_weights is None else [round(value, 2) for value in class_weights.tolist()],
    )
    logger.info("label_smoothing={}", args.label_smoothing)
    logger.info("pretrained_weights={} mlp_head={}", args.pretrained_weights, args.mlp_head)
    if args.llrd_decay != 1.0:
        head_params = list(model.classifier.parameters()) + list(
            model.image_encoder.model.encoder.ln.parameters()
        )
        parameter_groups = [{"params": head_params, "lr": args.lr}]
        unfrozen_blocks = list(model.image_encoder.model.encoder.layers)[-args.trainable_blocks:]
        for block_index, block in enumerate(unfrozen_blocks):
            depth = len(unfrozen_blocks) - block_index
            parameter_groups.append(
                {"params": list(block.parameters()), "lr": args.lr * (args.llrd_decay ** depth)}
            )
        logger.info(
            "llrd_decay={} group_lrs={}",
            args.llrd_decay,
            [round(group["lr"], 8) for group in parameter_groups],
        )
    else:
        parameter_groups = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=1e-4)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    class_names = [name for name, _ in sorted(CATEGORIES.items(), key=lambda item: item[1])]
    best_score = -float("inf")
    best_validation_loss = float("inf")
    patience_left = args.early_stop_patience
    for epoch in range(1, args.epochs + 1):
        if args.scheduler != "none":
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * learning_rate_scale(args, epoch - 1)
        train_loss, train_accuracy, _, _ = run_epoch(
            model, train_loader, optimizer, device, class_weights, f"epoch {epoch}/{args.epochs} train",
            label_smoothing=args.label_smoothing, mixup_alpha=args.mixup_alpha,
        )
        with torch.no_grad():
            validation_loss, validation_accuracy, validation_predictions, validation_labels = run_epoch(
                model, validation_loader, None, device, class_weights, f"epoch {epoch}/{args.epochs} val"
            )
        validation_metrics = multiclass_metrics(validation_predictions, validation_labels)
        logger.info(
            "epoch={} train_loss={:.4f} train_acc={:.4f} val_loss={:.4f} val_acc={:.4f} "
            "val_macro_f1={:.4f} val_bal_acc={:.4f} val_recall={} lr={:.2e}",
            epoch,
            train_loss,
            train_accuracy,
            validation_loss,
            validation_accuracy,
            validation_metrics["macro_f1"],
            validation_metrics["balanced_accuracy"],
            [round(value, 3) for value in validation_metrics["recall"]],
            optimizer.param_groups[0]["lr"],
        )
        score = validation_accuracy if args.select_by == "val_acc" else -validation_loss
        if score > best_score:
            best_score = score
            encoder_state = model.image_encoder.state_dict()
            if args.img_size != 224:
                encoder_state["model.encoder.pos_embedding"] = interpolate_pos_embedding(
                    encoder_state["model.encoder.pos_embedding"], args.img_size // 16, 14
                ).detach()
            torch.save(
                {
                    "image_encoder": encoder_state,
                    "classifier": model.classifier.state_dict(),
                    "class_count": 4,
                    "epoch": epoch,
                },
                output / "checkpoint.pt",
            )
        if args.early_stop_patience > 0:
            if validation_loss < best_validation_loss - 1e-4:
                best_validation_loss = validation_loss
                patience_left = args.early_stop_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info(
                        "early stop at epoch={} best_val_loss={:.4f}", epoch, best_validation_loss
                    )
                    break
    checkpoint = torch.load(output / "checkpoint.pt", map_location=device)
    encoder_state = checkpoint["image_encoder"]
    if args.img_size != 224:
        encoder_state = dict(encoder_state)
        encoder_state["model.encoder.pos_embedding"] = interpolate_pos_embedding(
            encoder_state["model.encoder.pos_embedding"], 14, args.img_size // 16
        )
    model.image_encoder.load_state_dict(encoder_state)
    model.classifier.load_state_dict(checkpoint["classifier"])
    with torch.no_grad():
        test_loss, test_accuracy, test_predictions, test_labels = run_epoch(
            model, test_loader, None, device, class_weights, "test"
        )
    test_metrics = multiclass_metrics(test_predictions, test_labels)
    logger.info("test_loss={:.4f} test_acc={:.4f}", test_loss, test_accuracy)
    logger.info(
        "test_macro_f1={:.4f} test_bal_acc={:.4f}",
        test_metrics["macro_f1"],
        test_metrics["balanced_accuracy"],
    )
    logger.info(
        "class_order={} test_support={} test_recall={}",
        class_names,
        test_metrics["support"],
        [round(value, 3) for value in test_metrics["recall"]],
    )
    logger.info("test_confusion_matrix={}", test_metrics["confusion_matrix"])
    if args.img_size != 224:
        compat_set = ZenodoImageDataset(
            args.csv, args.images, test_indices, train=False, cache_path=args.cache, crop_size=224
        )
        compat_loader = DataLoader(compat_set, batch_size=args.batch_size, **loader_kwargs)
        compat_model = ZenodoViTClassifier(
            pretrained=False, trainable_blocks=args.trainable_blocks
        ).to(device)
        compat_model.image_encoder.load_state_dict(checkpoint["image_encoder"])
        compat_model.classifier.load_state_dict(checkpoint["classifier"])
        compat_model.eval()
        with torch.no_grad():
            compat_loss, compat_accuracy, _, _ = run_epoch(
                compat_model, compat_loader, None, device, class_weights, "test-224-compat"
            )
        logger.info("test_224_loss={:.4f} test_224_acc={:.4f}", compat_loss, compat_accuracy)


if __name__ == "__main__":
    main()
