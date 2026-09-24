import csv
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import make_image_transform


MISSING = {"", "NAN", "nan", "NA", "N/A", "null", "NULL", "/", "#VALUE!"}
LABELS = {
    "level1_mild_or_normal": 0,
    "level2_moderate": 1,
    "level3_severe": 2,
}


@dataclass
class SysuRecord:
    sample_id: str
    label: int
    image_path: str
    sex: float | None
    age: float | None
    smoking: float | None
    smoking_years: float | None
    smoking_amount: float | None
    alcohol: float | None
    alcohol_years: float | None
    alcohol_amount: float | None
    betel: float | None
    betel_years: float | None
    betel_amount: float | None
    tct: str
    dna: float | None
    methylation: float | None
    tct_id: int = 0


def _text(value: str) -> str:
    return value.strip()


def _number(value: str) -> float | None:
    value = _text(value)
    if value in MISSING:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _binary(value: str) -> float | None:
    value = _text(value)
    if value in MISSING:
        return None
    if value in {"是", "有", "1", "yes", "Yes", "Y"}:
        return 1.0
    if value in {"否", "无", "0", "no", "No", "N"}:
        return 0.0
    return _number(value)


def _sex(value: str) -> float | None:
    value = _text(value)
    if value in MISSING:
        return None
    if value in {"男", "M", "m"}:
        return 1.0
    if value in {"女", "F", "f"}:
        return 0.0
    return None


def build_tct_vocab(records: list[SysuRecord]) -> dict[str, int]:
    values = sorted({record.tct for record in records if record.tct})
    return {value: index for index, value in enumerate(values, start=1)}


def load_sysu_records(csv_path: str | Path) -> list[SysuRecord]:
    records = []
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            records.append(
                SysuRecord(
                    sample_id=row["sample_id"],
                    label=LABELS[row["label"]],
                    image_path=row["image_path"],
                    sex=_sex(row["sex"]),
                    age=_number(row["age"]),
                    smoking=_binary(row["是否吸烟"]),
                    smoking_years=_number(row["吸烟年限（年）"]),
                    smoking_amount=_number(row["吸烟量（支/天）"]),
                    alcohol=_binary(row["是否饮酒"]),
                    alcohol_years=_number(row["饮酒年限(年)"]),
                    alcohol_amount=_number(row["饮酒量(ml/天)"]),
                    betel=_binary(row["是否嚼槟榔"]),
                    betel_years=_number(row["嚼槟榔年限(年)"]),
                    betel_amount=_number(row["嚼槟榔量（颗/天）"]),
                    tct=_text(row["TCT意见"]),
                    dna=_number(row["DNA倍体分析结果"]),
                    methylation=_number(row["甲基化总数"]),
                )
            )
    vocab = build_tct_vocab(records)
    for record in records:
        record.tct_id = vocab.get(record.tct, 0)
    return records


def _value_mask(values: list[float | None], scale: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    data = []
    mask = []
    for value, divisor in zip(values, scale):
        data.append(0.0 if value is None else value / divisor)
        mask.append(0.0 if value is None else 1.0)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


class SysuDataset(Dataset):
    def __init__(self, records, image_root: str | Path, train: bool = False):
        self.records = records
        self.image_root = Path(image_root)
        self.transform = make_image_transform(train)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        demo, demo_mask = _value_mask([record.sex, record.age], [1.0, 100.0])
        history, history_mask = _value_mask(
            [
                record.smoking,
                record.smoking_years,
                record.smoking_amount,
                record.alcohol,
                record.alcohol_years,
                record.alcohol_amount,
                record.betel,
                record.betel_years,
                record.betel_amount,
            ],
            [1.0, 50.0, 100.0, 1.0, 50.0, 1000.0, 1.0, 20.0, 50.0],
        )
        dna, dna_mask = _value_mask([record.dna], [1.0])
        methylation, methylation_mask = _value_mask([record.methylation], [10.0])

        image_present = bool(record.image_path)
        image = torch.zeros(3, 224, 224)
        if image_present:
            image_file = self.image_root / record.image_path
            image = self.transform(Image.open(image_file).convert("RGB"))

        return {
            "image": image,
            "image_present": torch.tensor(image_present),
            "demo": demo,
            "demo_mask": demo_mask,
            "history": history,
            "history_mask": history_mask,
            "tct": torch.tensor(record.tct_id, dtype=torch.long),
            "tct_present": torch.tensor(bool(record.tct)),
            "dna": dna,
            "dna_mask": dna_mask.bool(),
            "methylation": methylation,
            "methylation_mask": methylation_mask.bool(),
            "label": torch.tensor(record.label, dtype=torch.long),
            "sample_id": record.sample_id,
        }


def split_sysu_records(records: list[SysuRecord], validation_fraction=0.15, seed=42):
    generator = torch.Generator().manual_seed(seed)
    train, validation = [], []
    for label in sorted({record.label for record in records}):
        group = [record for record in records if record.label == label]
        order = torch.randperm(len(group), generator=generator).tolist()
        validation_count = max(1, round(len(group) * validation_fraction))
        validation.extend(group[index] for index in order[:validation_count])
        train.extend(group[index] for index in order[validation_count:])
    return train, validation
