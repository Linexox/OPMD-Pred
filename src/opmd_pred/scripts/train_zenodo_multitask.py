"""多任务训练：共享 ViT 编码器，主头 4 分类(coarse) + 辅头精细诊断(fine)。

-  推理时只需图像，仅用 coarse 头 → 与学长"image_encoder + classifier"交付兼容。
-  辅头用精细 Clinical Diagnosis(≈149 类)做辅助监督，梯度沿临床实体边界拉特征，
   目标提升 coarse 4 分类 accuracy(尤其 Benign↔OPMD 混淆)。
-  v5 配方复刻：lr1e-4 / 15ep / batch256 / trainable-blocks6 / inverse 权重 / 288缓存+裁剪。
"""
import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

from opmd_pred.data.zenodo import CATEGORIES, ZenodoImageDataset, _load_rows, split_stratified
from opmd_pred.model.vit import ImageEncoder
from opmd_pred.scripts.common import configure_logger, device_name, multiclass_metrics, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/zenodo/Imagewise_Data.csv")
    parser.add_argument("--images", default="data/zenodo/Images")
    parser.add_argument("--output", default="outputs/zenodo_multitask")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--trainable-blocks", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cache", default="outputs/zenodo_cache288.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--class-weight", choices=["inverse", "sqrt", "none"], default="inverse")
    parser.add_argument("--fine-lambda", type=float, default=0.5)
    parser.add_argument("--fine-min-count", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=224)
    return parser.parse_args()


class ZenodoFineDataset(ZenodoImageDataset):
    """在 ZenodoImageDataset 上叠一片细分类标签，fine_codes 与实例顺序对齐。"""

    def __init__(self, *args, fine_codes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fine_codes = fine_codes

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.fine_codes is not None:
            item["fine"] = self.fine_codes[index]
        return item


class MultitaskClassifier(torch.nn.Module):
    def __init__(self, class_count=4, fine_count=None, trainable_blocks=6, weights_name="v1", img_size=224):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=True, weights_name=weights_name, img_size=img_size)
        self.image_encoder.unfreeze_last_blocks(trainable_blocks)
        self.coarse_head = torch.nn.Linear(self.image_encoder.output_dim, class_count)
        self.fine_head = (
            torch.nn.Linear(self.image_encoder.output_dim, fine_count) if fine_count else None
        )

    def forward(self, image):
        features = self.image_encoder(image)
        coarse = self.coarse_head(features)
        fine = self.fine_head(features) if self.fine_head is not None else None
        return coarse, fine


def build_fine(csv_path, fine_min_count):
    """返回 (全局index->细编码list, 细类别数)。fine_index 与 _load_rows/csv 行顺序一致。"""
    rows = _load_rows(csv_path)
    counts = {}
    for row in rows:
        name = (row.get("Clinical Diagnosis") or "").strip()
        counts[name] = counts.get(name, 0) + 1
    keep = {name for name, c in counts.items() if c >= fine_min_count}
    index_by_name = {}
    for name, c in counts.items():
        if name in keep:
            index_by_name[name] = len(index_by_name)
    other = len(index_by_name)
    fine = []
    for row in rows:
        name = (row.get("Clinical Diagnosis") or "").strip()
        fine.append(index_by_name.get(name, other))
    n_total = other + 1 if counts else 1
    return fine, n_total


def run_epoch(model, loader, optimizer, device, coarse_weights, fine_weights, fine_lambda, desc):
    training = optimizer is not None
    model.train(training)
    total_loss = total_coarse_loss = 0.0
    total_correct = total_count = 0
    predictions, labels = [], []
    progress = tqdm(loader, desc=desc, unit="batch")
    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        coarse, fine = model(image)
        coarse_loss = torch.nn.functional.cross_entropy(
            coarse, label, weight=coarse_weights
        )
        loss = coarse_loss
        if fine is not None:
            fine_label = batch["fine"].to(device, non_blocking=True)
            fine_loss = torch.nn.functional.cross_entropy(fine, fine_label, weight=fine_weights)
            loss = loss + fine_lambda * fine_loss
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        prediction = coarse.argmax(1)
        total_loss += loss.item() * label.size(0)
        total_coarse_loss += coarse_loss.item() * label.size(0)
        total_correct += (prediction == label).sum().item()
        total_count += label.size(0)
        predictions.append(prediction.detach().cpu())
        labels.append(label.detach().cpu())
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return (
        total_loss / total_count,
        total_coarse_loss / total_count,
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

    fine, fine_count = build_fine(args.csv, args.fine_min_count)
    train_indices, validation_indices, test_indices = split_stratified(
        args.csv, seed=args.split_seed
    )
    logger.info(
        "split=image split_seed={} train={} val={} test={} fine_count={}",
        args.split_seed, len(train_indices), len(validation_indices), len(test_indices), fine_count,
    )
    train_set = ZenodoFineDataset(
        args.csv, args.images, train_indices, train=True, cache_path=args.cache, crop_size=args.img_size,
        fine_codes=[fine[i] for i in train_indices],
    )
    validation_set = ZenodoFineDataset(
        args.csv, args.images, validation_indices, train=False, cache_path=args.cache, crop_size=args.img_size,
        fine_codes=[fine[i] for i in validation_indices],
    )
    test_set = ZenodoFineDataset(
        args.csv, args.images, test_indices, train=False, cache_path=args.cache, crop_size=args.img_size,
        fine_codes=[fine[i] for i in test_indices],
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, **loader_kwargs)

    model = MultitaskClassifier(
        class_count=len(CATEGORIES), fine_count=fine_count,
        trainable_blocks=args.trainable_blocks, img_size=args.img_size,
    ).to(device)
    counts = torch.bincount(
        torch.tensor([CATEGORIES[row["Category"]] for row in train_set.rows]),
        minlength=len(CATEGORIES),
    ).float()
    if args.class_weight == "none":
        coarse_weights = None
    else:
        coarse_weights = counts.sum() / counts
        if args.class_weight == "sqrt":
            coarse_weights = coarse_weights.sqrt()
        coarse_weights = coarse_weights.to(device)
    fine_counts = torch.tensor(
        [sum(1 for i in train_indices if fine[i] == c) for c in range(fine_count)], dtype=torch.float
    )
    fine_weights = (fine_counts.sum() / (fine_counts + 1e-6)).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    class_names = [name for name, _ in sorted(CATEGORIES.items(), key=lambda item: item[1])]
    best_coarse_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_coarse, train_acc, _, _ = run_epoch(
            model, train_loader, optimizer, device, coarse_weights, fine_weights, args.fine_lambda,
            f"epoch {epoch}/{args.epochs} train",
        )
        with torch.no_grad():
            val_loss, val_coarse, val_acc, val_pred, val_label = run_epoch(
                model, validation_loader, None, device, coarse_weights, fine_weights, args.fine_lambda,
                f"epoch {epoch}/{args.epochs} val",
            )
        metrics = multiclass_metrics(val_pred, val_label)
        logger.info(
            "epoch={} train_loss={:.4f} val_loss={:.4f} val_coarse_acc={:.4f} "
            "val_macro_f1={:.4f} val_bal_acc={:.4f} val_recall={}",
            epoch, train_loss, val_loss, val_acc, metrics["macro_f1"],
            metrics["balanced_accuracy"], [round(v, 3) for v in metrics["recall"]],
        )
        if val_acc > best_coarse_acc:
            best_coarse_acc = val_acc
            torch.save({
                "image_encoder": model.image_encoder.state_dict(),
                "coarse_head": model.coarse_head.state_dict(),
                "class_count": len(CATEGORIES),
                "epoch": epoch,
            }, output / "checkpoint.pt")
    ckpt = torch.load(output / "checkpoint.pt")
    model.image_encoder.load_state_dict(ckpt["image_encoder"])
    model.coarse_head.load_state_dict(ckpt["coarse_head"])
    model.eval()
    with torch.no_grad():
        _, _, test_acc, test_pred, test_label = run_epoch(
            model, test_loader, None, device, coarse_weights, fine_weights, args.fine_lambda, "test"
        )
    metrics = multiclass_metrics(test_pred, test_label)
    logger.info("test_acc={:.4f}  test_macro_f1={:.4f}  test_bal_acc={:.4f}", test_acc, metrics["macro_f1"], metrics["balanced_accuracy"])
    logger.info(
        "test_class_order={} test_support={} test_recall={}", class_names, metrics["support"],
        [round(v, 3) for v in metrics["recall"]],
    )
    logger.info("test_confusion_matrix={}", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()