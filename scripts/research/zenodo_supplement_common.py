"""Shared artifact, image and model contracts for the planned Zenodo study."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16

from opmd_pred.data.sysu_protocol import file_sha256
from opmd_pred.model.resnet import ResNet18Classifier, PREPROCESSING


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def save(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def configure(seed):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_rows(config):
    path = Path(config["records"])
    audit = read(path.with_name("annotation_audit.json"))
    if file_sha256(path) != audit["images_jsonl_sha256"]:
        raise ValueError("Prepared records changed after audit")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 2994:
        raise ValueError("Wrong fixed Zenodo scope")
    return rows


class PhotoClassifier(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super().__init__()
        self.backbone = backbone
        if backbone == "resnet18":
            m = ResNet18Classifier(pretrained=pretrained)
            self.encoder, self.classifier = m.encoder, m.classifier
            self.feature_dim = 512
        elif backbone == "vit_b16":
            self.encoder = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            self.encoder.heads = nn.Identity()
            self.feature_dim = 768
            self.classifier = nn.Linear(self.feature_dim, 4)
        else:
            raise ValueError("Unsupported classification backbone")

    def forward(self, image):
        return self.classifier(self.encoder(image))


def metrics(p, labels):
    y = torch.as_tensor(labels, dtype=torch.long)
    predicted = torch.as_tensor(p).argmax(1)
    cm = torch.bincount(y*4+predicted, minlength=16).reshape(4, 4)
    tp = cm.diag().double()
    recalls = tp / cm.sum(1).clamp_min(1)
    f1 = 2*tp / (cm.sum(0)+cm.sum(1)).clamp_min(1)
    return {"accuracy": float(tp.sum()/cm.sum()), "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recalls.mean()), "class_recall": recalls.tolist(),
        "class_support": cm.sum(1).tolist(), "confusion_matrix": cm.tolist(), "count": len(y)}


def pad224(image):
    return ImageOps.pad(image, (224, 224), method=Image.Resampling.BILINEAR,
                        color=(124, 116, 104), centering=(.5, .5))


def roi_crop(image, row):
    if not row["roi_valid"]:
        return image
    boxes = row["boxes"]
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    dx, dy = .2*(x2-x1), .2*(y2-y1)
    return image.crop((max(0, x1-dx), max(0, y1-dy), min(image.width, x2+dx), min(image.height, y2+dy)))


@lru_cache(maxsize=4)
def source_paths(root):
    mapping={}
    for p in Path(root).rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}:
            if p.stem in mapping:raise ValueError('Ambiguous photo stem')
            mapping[p.stem]=p
    return mapping


def image_path(config,row):
    # Local source has an extra Images/ level; remote source is flat. SHA pins
    # bytes, not a machine-specific directory layout.
    return source_paths(str(config['images']))[row['image_name']]


def pixel_cache(config, rows, roi=False):
    root = Path(config["records"]).parent / "pixel_cache"
    root.mkdir(exist_ok=True)
    identity = file_sha256(config["records"])
    name = "roi224" if roi else "global224"
    path = root / (name + "-" + identity[:16] + ".npy")
    receipt = path.with_suffix(".json")
    if path.exists() and receipt.exists():
        saved = read(receipt)
        if saved["records_sha256"] != identity or saved["cache_sha256"] != file_sha256(path):
            raise ValueError("Pixel cache identity mismatch")
        return np.load(path, mmap_mode="r")
    temp = path.with_name(path.name + ".partial")
    pixels = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint8, shape=(len(rows), 224, 224, 3))
    for index, row in enumerate(rows):
        source = image_path(config,row)
        if file_sha256(source) != row["sha256"]:
            raise ValueError("Source photo bytes changed")
        with Image.open(source) as image:
            image = image.convert("RGB")
            if roi:
                image = roi_crop(image, row)
            pixels[index] = np.asarray(pad224(image))
    pixels.flush()
    del pixels
    temp.replace(path)
    write(receipt, {"records_sha256": identity, "cache_sha256": file_sha256(path), "shape": [len(rows),224,224,3]})
    return np.load(path, mmap_mode="r")


class ClassificationPhotos(torch.utils.data.Dataset):
    def __init__(self, config, rows, split, training=False, views=None):
        self.all_rows = rows
        self.indices = [i for i, r in enumerate(rows) if r["split"] == split]
        self.rows = [rows[i] for i in self.indices]
        self.pixels = pixel_cache(config, rows)
        self.training, self.seed, self.epoch = training, config["seed"], 0
        self.views = views or config["views"]
        self.roi = pixel_cache(config, rows, True) if self.views == "manual_roi" else None
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomRotation(10, fill=(124,116,104)),
            transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.05),
            transforms.ToTensor(), transforms.Normalize(PREPROCESSING["mean"], PREPROCESSING["std"])])
        self.evaluation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(PREPROCESSING["mean"], PREPROCESSING["std"])])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        count = 2 if self.training and self.views != "global" else 1
        values = []
        for view in range(count):
            source = self.roi if view == 1 and self.roi is not None else self.pixels
            image = Image.fromarray(source[index])
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.seed + self.epoch*100003 + index*5 + view)
                values.append(self.transform(image) if self.training else self.evaluation(image))
        return torch.stack(values), self.all_rows[index]["label"], index


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    model.eval()
    ps, ys, indices = [], [], []
    for image, label, ix in loader:
        ps.append(model(image[:, 0].to(device)).softmax(1).cpu())
        ys.extend(label.tolist())
        indices.extend(ix.tolist())
    probabilities = torch.cat(ps)
    return probabilities, ys, indices, metrics(probabilities, ys)
