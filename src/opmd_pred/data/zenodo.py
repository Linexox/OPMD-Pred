import csv
import random
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from loguru import logger

from .transforms import make_cached_image_transform, make_image_transform


CATEGORIES = {"Benign": 0, "Healthy": 1, "OPMD": 2, "OCA": 3}
ImageFile.LOAD_TRUNCATED_IMAGES = True
_CACHE_STORE = {}


def _load_rows(csv_path):
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    valid_rows = []
    skipped = 0
    for row in rows:
        category = row["Category"].strip()
        if category not in CATEGORIES:
            skipped += 1
            continue
        row["Category"] = category
        valid_rows.append(row)
    if skipped:
        logger.warning("Skipped {} Zenodo rows with missing or unknown Category", skipped)
    return valid_rows


def _patient_ids(csv_path):
    patient_csv = Path(csv_path).with_name("Patientwise_Data.csv")
    with patient_csv.open(encoding="utf-8-sig", newline="") as file:
        return [row["Patient ID"] for row in csv.DictReader(file)]


def _attach_patient_ids(rows, patient_ids):
    for row in rows:
        matches = [patient for patient in patient_ids if row["Image Name"].startswith(patient + "-")]
        row["Patient ID"] = max(matches, key=len)


class ZenodoImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        indices=None,
        train=False,
        cache_path: str | Path | None = None,
    ):
        rows = _load_rows(csv_path)
        _attach_patient_ids(rows, _patient_ids(csv_path))
        self.rows = rows if indices is None else [rows[index] for index in indices]
        self.cache = None
        self.cache_indices = None
        if cache_path:
            cache_path = str(Path(cache_path))
            if cache_path not in _CACHE_STORE:
                _CACHE_STORE[cache_path] = torch.load(cache_path, map_location="cpu")
            self.cache = _CACHE_STORE[cache_path]
            self.cache_indices = self.cache["indices"]
            self.transform = make_cached_image_transform(train)
            return
        self.image_root = Path(image_root)
        self.transform = make_image_transform(train)
        self.images = {
            path.stem: path
            for path in self.image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_name = row["Image Name"]
        if self.cache is None:
            image = Image.open(self.images[image_name]).convert("RGB")
        else:
            image = self.cache["images"][self.cache_indices[image_name]]
        return {
            "image": self.transform(image),
            "label": torch.tensor(CATEGORIES[row["Category"]], dtype=torch.long),
            "patient_id": row["Patient ID"],
        }


def split_by_patient(csv_path: str | Path, validation_fraction=0.1, test_fraction=0.1, seed=42):
    rows = _load_rows(csv_path)
    _attach_patient_ids(rows, _patient_ids(csv_path))
    patients = sorted({row["Patient ID"] for row in rows})
    random.Random(seed).shuffle(patients)
    test_count = round(len(patients) * test_fraction)
    validation_count = round(len(patients) * validation_fraction)
    test_patients = set(patients[:test_count])
    validation_patients = set(patients[test_count : test_count + validation_count])
    held_out = test_patients | validation_patients
    train = [index for index, row in enumerate(rows) if row["Patient ID"] not in held_out]
    validation = [index for index, row in enumerate(rows) if row["Patient ID"] in validation_patients]
    test = [index for index, row in enumerate(rows) if row["Patient ID"] in test_patients]
    return train, validation, test
