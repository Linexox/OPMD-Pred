import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import make_image_transform


CATEGORIES = {"Benign": 0, "Healthy": 1, "OPMD": 2, "OCA": 3}


def _patient_ids(csv_path):
    patient_csv = Path(csv_path).with_name("Patientwise_Data.csv")
    with patient_csv.open(encoding="utf-8-sig", newline="") as file:
        return [row["Patient ID"] for row in csv.DictReader(file)]


def _attach_patient_ids(rows, patient_ids):
    for row in rows:
        matches = [patient for patient in patient_ids if row["Image Name"].startswith(patient + "-")]
        row["Patient ID"] = max(matches, key=len)


class ZenodoImageDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_root: str | Path, indices=None, train=False):
        with Path(csv_path).open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        _attach_patient_ids(rows, _patient_ids(csv_path))
        self.rows = rows if indices is None else [rows[index] for index in indices]
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
        image = Image.open(self.images[image_name]).convert("RGB")
        return {
            "image": self.transform(image),
            "label": torch.tensor(CATEGORIES[row["Category"]], dtype=torch.long),
            "patient_id": row["Patient ID"],
        }


def split_by_patient(csv_path: str | Path, validation_fraction=0.1, test_fraction=0.1, seed=42):
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
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
