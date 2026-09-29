import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

from opmd_pred.data.zenodo import (
    CATEGORIES,
    ZenodoImageDataset,
    split_by_patient,
    split_stratified,
)
from opmd_pred.model.vit import ZenodoViTClassifier
from opmd_pred.scripts.common import configure_logger, device_name, multiclass_metrics, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--csv", default="data/zenodo/Imagewise_Data.csv")
    parser.add_argument("--images", default="data/zenodo/Images")
    parser.add_argument("--cache", default="outputs/zenodo_cache288.pt")
    parser.add_argument("--output", default="outputs/zenodo_ensemble")
    parser.add_argument("--trainable-blocks", type=int, default=6)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["patient", "image"], default="patient")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tta", choices=["none", "hflip"], default="hflip")
    return parser.parse_args()


def average_accuracy(probability, labels):
    return (probability.argmax(dim=1) == labels).float().mean().item()


def main():
    args = parse_args()
    set_seed(args.split_seed)
    configure_logger(Path(args.output))
    device = torch.device(device_name())
    if args.split_mode == "patient":
        _, _, test_indices = split_by_patient(args.csv, seed=args.split_seed)
    else:
        _, _, test_indices = split_stratified(args.csv, seed=args.split_seed)
    test_set = ZenodoImageDataset(
        args.csv, args.images, test_indices, train=False, cache_path=args.cache
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    models = []
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location=device)
        mlp_head = any(key.startswith("0.") for key in checkpoint["classifier"])
        is_dinov2 = any(key.startswith("model.blocks.") for key in checkpoint["image_encoder"])
        model = ZenodoViTClassifier(
            pretrained=False,
            trainable_blocks=args.trainable_blocks,
            weights_name="dinov2_vitb14" if is_dinov2 else "v1",
            mlp_head=mlp_head,
        ).to(device)
        model.image_encoder.load_state_dict(checkpoint["image_encoder"])
        model.classifier.load_state_dict(checkpoint["classifier"])
        model.eval()
        models.append(model)

    model_probabilities = []
    label_batches = []
    for model_index, model in enumerate(models):
        batches = []
        progress = tqdm(test_loader, desc="evaluating", unit="batch")
        for batch in progress:
            image = batch["image"].to(device, non_blocking=True)
            with torch.no_grad():
                probability = model(image).softmax(dim=1)
                if args.tta == "hflip":
                    flipped = torch.flip(image, dims=[3])
                    probability = probability + model(flipped).softmax(dim=1)
            batches.append(probability.cpu())
            if model_index == 0:
                label_batches.append(batch["label"])
        model_probabilities.append(torch.cat(batches))

    labels = torch.cat(label_batches)
    class_names = [name for name, _ in sorted(CATEGORIES.items(), key=lambda item: item[1])]
    ensemble_probability = torch.stack(model_probabilities).mean(dim=0)

    logger.info("checkpoints={}", args.checkpoints)
    logger.info("tta={} test_size={}", args.tta, len(labels))
    single_accuracies = []
    for index, probability in enumerate(model_probabilities):
        metrics = multiclass_metrics(probability.argmax(dim=1), labels)
        accuracy = average_accuracy(probability, labels)
        single_accuracies.append(accuracy)
        logger.info(
            "model[{}]={} acc={:.4f} macro_f1={:.4f} bal_acc={:.4f}",
            index,
            args.checkpoints[index],
            accuracy,
            metrics["macro_f1"],
            metrics["balanced_accuracy"],
        )
    metrics = multiclass_metrics(ensemble_probability.argmax(dim=1), labels)
    logger.info(
        "single_mean_acc={:.4f} ensemble_acc={:.4f} gain={:+.4f}",
        sum(single_accuracies) / len(single_accuracies),
        average_accuracy(ensemble_probability, labels),
        average_accuracy(ensemble_probability, labels)
        - sum(single_accuracies) / len(single_accuracies),
    )
    logger.info(
        "ensemble_macro_f1={:.4f} ensemble_bal_acc={:.4f}",
        metrics["macro_f1"],
        metrics["balanced_accuracy"],
    )
    logger.info(
        "class_order={} test_support={} test_recall={}",
        class_names,
        metrics["support"],
        [round(value, 3) for value in metrics["recall"]],
    )
    logger.info("test_confusion_matrix={}", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()