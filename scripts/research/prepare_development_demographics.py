"""Extract age and sex for the frozen 117-record development cohort only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABEL_TO_ID = {
    "level1_mild_or_normal": 0,
    "level2_moderate": 1,
    "level3_severe": 2,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    entries = {item["sample_id"]: item for item in protocol["development_entries"]}
    if len(entries) != 117:
        raise ValueError("Expected the frozen 117-record development cohort")
    with args.source.open(encoding="utf-8-sig", newline="") as stream:
        source = {row["sample_id"]: row for row in csv.DictReader(stream)}
    selected = []
    for sample_id, entry in entries.items():
        row = source.get(sample_id)
        if row is None:
            raise ValueError(f"Missing demographic source row for {sample_id}")
        if LABEL_TO_ID[row["label"]] != entry["label"]:
            raise ValueError(f"Label differs for {sample_id}")
        age_text = row.get("age", "").strip()
        age = float(age_text) if age_text else None
        if age is not None and not 0 <= age <= 120:
            raise ValueError(f"Invalid age for {sample_id}")
        sex_text = row.get("sex", "").strip()
        if sex_text in {"男", "M", "m"}:
            sex = 1
        elif sex_text in {"女", "F", "f"}:
            sex = 0
        elif not sex_text:
            sex = None
        else:
            raise ValueError(f"Unknown sex value for {sample_id}")
        selected.append({"sample_id": sample_id, "label": entry["label"],
                         "age": "" if age is None else f"{age:g}",
                         "sex": "" if sex is None else str(sex)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "label", "age", "sex"))
        writer.writeheader()
        writer.writerows(selected)
    print(json.dumps({
        "records": len(selected),
        "age_available": sum(bool(row["age"]) for row in selected),
        "sex_available": sum(bool(row["sex"]) for row in selected),
        "both_available": sum(bool(row["age"] and row["sex"]) for row in selected),
        "locked_records_included": 0,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
