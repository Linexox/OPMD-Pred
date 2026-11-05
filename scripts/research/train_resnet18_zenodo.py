"""Train a classic photo encoder on the original, hash-pinned Zenodo split."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import time

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from opmd_pred.data.transforms import ResNetAspectPad, make_image_transform
from opmd_pred.data.zenodo import CATEGORIES
from opmd_pred.data.sysu_protocol import file_sha256
from opmd_pred.model.resnet import PREPROCESSING, ResNet18Classifier, ResNet18Encoder
from opmd_pred.scripts.common import set_seed


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def metrics(probabilities, labels):
    predicted = probabilities.argmax(1)
    cm = torch.bincount(labels * 4 + predicted, minlength=16).reshape(4, 4)
    tp = cm.diag().float()
    recall = tp / cm.sum(1).clamp_min(1)
    f1 = 2 * tp / (cm.sum(0) + cm.sum(1)).clamp_min(1)
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean()),
            "class_recall": recall.tolist(), "class_support": cm.sum(1).tolist(),
            "confusion_matrix": cm.tolist(), "count": len(labels)}


def audit(root, manifest):
    images = {}
    for path in sorted((root / "Images").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if path.stem in images:
                raise ValueError("Ambiguous image stem: " + path.stem)
            images[path.stem] = path
    with (root / "Imagewise_Data.csv").open(encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    valid = {r["Image Name"]: r["Category"].strip() for r in source_rows if r["Category"].strip() in CATEGORIES}
    with (root / "Patientwise_Data.csv").open(encoding="utf-8-sig", newline="") as stream:
        patients = [r["Patient ID"] for r in csv.DictReader(stream)]
    if manifest["category_order"] != CATEGORIES:
        raise ValueError("Category mapping changed")
    groups, hashes, names, counts = {}, {}, set(), {}
    for split, expected in (("train", 2366), ("validation", 315), ("test", 313)):
        rows = manifest["splits"][split]
        if len(rows) != expected:
            raise ValueError("Wrong original split size")
        groups[split], hashes[split] = set(), set()
        for row in rows:
            name = row["image_name"]
            if name in names or valid.get(name) != row["category"]:
                raise ValueError("Duplicate or incorrectly labelled image")
            matches = [p for p in patients if name.startswith(p + "-")]
            if not matches or max(matches, key=len) != row["patient_id"]:
                raise ValueError("Patient mapping differs from original split")
            digest = file_sha256(images[name])
            if digest != row["sha256"]:
                raise ValueError("Source image SHA mismatch: " + name)
            names.add(name)
            groups[split].add(row["patient_id"])
            hashes[split].add(digest)
        counts[split] = {"images": len(rows), "patients": len(groups[split]),
                         "categories": dict(Counter(r["category"] for r in rows))}
        if counts[split]["patients"] != manifest["counts"][split]["patients"]:
            raise ValueError("Patient count mismatch")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if groups[a] & groups[b] or hashes[a] & hashes[b]:
            raise ValueError("Cross-split patient or exact-image overlap")
    excluded = {r["image_name"] for r in manifest["excluded_cross_split_duplicates"]["train"]}
    if names != set(valid) - excluded or len(excluded) != 5:
        raise ValueError("Unexpected source row exclusions")
    return images, {"counts": counts, "source_hashes_checked": len(names),
                    "patient_and_exact_image_disjoint": True, "excluded_train_duplicates": sorted(excluded),
                    "source_csv_sha256": file_sha256(root / "Imagewise_Data.csv"),
                    "patient_csv_sha256": file_sha256(root / "Patientwise_Data.csv")}


class Photos(Dataset):
    def __init__(self, rows, images, training):
        self.rows = rows
        # Cache only padded pixels, never augmented training views or test-derived statistics.
        self.pixels = []
        for row in rows:
            with Image.open(images[row["image_name"]]) as image:
                self.pixels.append(ResNetAspectPad()(image.convert("RGB")))
        steps = []
        if training:
            steps = [transforms.RandomHorizontalFlip(),
                     transforms.RandomRotation(10, fill=(124, 116, 104)),
                     transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.05)]
        self.transform = transforms.Compose(steps + [transforms.ToTensor(),
            transforms.Normalize(PREPROCESSING["mean"], PREPROCESSING["std"])])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.transform(self.pixels[index]), CATEGORIES[self.rows[index]["category"]]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    for image, label in loader:
        probabilities.append(model(image.to(device)).softmax(1).cpu())
        labels.append(label)
    probabilities, labels = torch.cat(probabilities), torch.cat(labels)
    return probabilities, labels, metrics(probabilities, labels)


def implementation_check(model, sample, output, device):
    model.eval()
    sample = sample.unsqueeze(0).to(device)
    with torch.no_grad():
        initial = model(sample)
        batch_delta = float((model(sample.repeat(2, 1, 1, 1))[:1] - initial).abs().max())
    if batch_delta > 1e-4:
        raise ValueError("Batch/single prediction mismatch")
    frozen = ResNet18Encoder().to(device)
    frozen.load_state_dict(model.encoder.state_dict(), strict=True)
    frozen.freeze()
    before = {k: v.clone() for k, v in frozen.state_dict().items()}
    frozen.train()
    with torch.no_grad():
        frozen(sample)
    if frozen.training or any(not torch.equal(v, frozen.state_dict()[k]) for k, v in before.items()):
        raise ValueError("Frozen parameters or BatchNorm statistics changed")
    save(output / "imagenet_encoder.pt", model.encoder.artifact(source="ImageNet1K_V1"))
    loaded = ResNet18Encoder().to(device)
    loaded.load_artifact(output / "imagenet_encoder.pt")
    loaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(sample), model.encoder(sample), rtol=0, atol=0)
    return {"frozen_parameters_and_batchnorm_unchanged": True,
            "encoder_reload_exact": True, "batch_single_logits_max_absolute_error": batch_delta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; inspect it before choosing a fresh attempt")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images, audit_result = audit(args.data, manifest)
    write_json(args.output / "data-audit.json", audit_result)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(preprocessing=PREPROCESSING, feature_dim=512, category_order=CATEGORIES,
        backbone_lr=.0001, classifier_lr=.001, weight_decay=.0001, optimizer="AdamW",
        loss="unweighted_four_class_cross_entropy", all_encoder_layers_trainable=True,
        rotation_degrees=10, horizontal_flip_probability=.5,
        color_jitter={"brightness": .1, "contrast": .1, "saturation": .05},
        selection="validation accuracy, then macro_f1, then earlier epoch",
        manifest_sha256=file_sha256(args.manifest), script_sha256=file_sha256(__file__),
        torch_version=str(torch.__version__), torchvision_version=__import__("torchvision").__version__,
        tf32=False, cudnn_deterministic=True,
        source_sha256={str(p): file_sha256(p) for p in
            (Path(__file__), Path(__import__("opmd_pred.model.resnet", fromlist=["_"]).__file__))})
    write_json(args.output / "config.json", config)
    model = ResNet18Classifier(pretrained=True).to(args.device)
    with Image.open(images[manifest["splits"]["validation"][0]["image_name"]]) as image:
        sample = make_image_transform(False, "resnet18")(image.convert("RGB"))
    checks = implementation_check(model, sample, args.output, args.device)
    write_json(args.output / "implementation-checks.json", checks)
    datasets = {split: Photos(manifest["splits"][split], images, split == "train")
                for split in ("train", "validation")}
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
        num_workers=args.workers, pin_memory=args.device == "cuda", persistent_workers=args.workers > 0)
        for split, dataset in datasets.items()}
    # Checks should not consume randomness belonging to the actual optimization.
    set_seed(args.seed)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": .0001},
        {"params": model.classifier.parameters(), "lr": .001}], weight_decay=.0001)
    best_key, best_epoch, stale, history = (-1., -1.), 0, 0, []
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        loss_sum, count = 0., 0
        for image, label in loaders["train"]:
            image, label = image.to(args.device), label.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(image), label)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(label)
            count += len(label)
        _, _, validation = evaluate(model, loaders["validation"], args.device)
        item = {"epoch": epoch, "training_loss": loss_sum / count,
                "validation": validation, "seconds": time.monotonic() - started}
        history.append(item)
        key = (validation["accuracy"], validation["macro_f1"])
        if key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            save(args.output / "best.pt", {"format": "opmd-resnet18-four-class-v1",
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": config, "epoch": epoch, "validation": validation})
        else:
            stale += 1
        save(args.output / "last.pt", {"state": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "history": history, "best_epoch": best_epoch, "stale": stale,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if args.device == "cuda" else None,
            "python_rng": __import__("random").getstate(), "config": config})
        write_json(args.output / "history.json", history)
        print(json.dumps(item), flush=True)
        if stale >= args.patience:
            break
    best = torch.load(args.output / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(best["state"], strict=True)
    validation_p, validation_y, validation_m = evaluate(model, loaders["validation"], args.device)
    if validation_m != best["validation"]:
        raise ValueError("Reloaded checkpoint validation metrics changed")
    save(args.output / "zenodo_encoder.pt", model.encoder.artifact(
        source="ImageNet1K_V1_then_Zenodo_four_class", selected_epoch=best_epoch,
        manifest_sha256=config["manifest_sha256"], classifier_sha256=file_sha256(args.output / "best.pt")))
    reload_model = ResNet18Classifier().to(args.device)
    reload_model.load_state_dict(best["state"], strict=True)
    repeat_p, _, _ = evaluate(reload_model, loaders["validation"], args.device)
    torch.testing.assert_close(repeat_p, validation_p, rtol=0, atol=0)
    test = Photos(manifest["splits"]["test"], images, False)
    test_p, test_y, test_m = evaluate(model, DataLoader(test, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=args.device == "cuda"), args.device)
    for name, probabilities, labels in (("validation", validation_p, validation_y), ("test", test_p, test_y)):
        write_json(args.output / f"{name}-predictions.json", [dict(row, label=int(y), probabilities=p.tolist())
            for row, y, p in zip(manifest["splits"][name], labels, probabilities)])
    checks.update(full_classifier_reload_exact=True, best_validation_replayed=True)
    write_json(args.output / "implementation-checks.json", checks)
    write_json(args.output / "results.json", {"best_epoch": best_epoch, "epochs_run": len(history),
        "validation": validation_m, "test": test_m, "test_is_previously_used_internal_benchmark": True,
        "encoder_sha256": file_sha256(args.output / "zenodo_encoder.pt"),
        "imagenet_encoder_sha256": file_sha256(args.output / "imagenet_encoder.pt")})
    print(json.dumps({"completed": True, "test": test_m}), flush=True)


if __name__ == "__main__":
    main()
