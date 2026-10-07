"""Reviewed patient mapping and immutable SYSU split protocol; no patient-ID inference."""

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .sysu import LABELS, MODALITIES, available_modalities, load_sysu_records


PROTOCOL = "sysu-patient-hash-v1"
SPLITS = ("train", "val", "test")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_digest(manifest):
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def inspect_records(records, images):
    by_id = {record.sample_id: record for record in records}
    if len(by_id) != len(records) or "" in by_id:
        raise ValueError("sample_id must be unique and nonempty")
    hashes, labels_by_hash, excluded = {}, {}, []
    for record in records:
        if not any(available_modalities(record).values()):
            excluded.append({"sample_id": record.sample_id, "reason": "all_five_modalities_missing"})
        if record.image_path:
            image_file = Path(images) / record.image_path
            if not image_file.is_file():
                raise ValueError(f"Image file missing for sample_id={record.sample_id}")
            digest = file_sha256(image_file)
            hashes[record.sample_id] = digest
            labels_by_hash.setdefault(digest, {}).setdefault(record.label, []).append(record.sample_id)
        else:
            hashes[record.sample_id] = None
    conflicts = [groups for groups in labels_by_hash.values() if len(groups) > 1]
    if conflicts:
        raise ValueError(f"Identical image has conflicting labels; review these sample IDs before splitting: {conflicts}")
    return by_id, hashes, excluded


def build_manifest(csv_path, images, groups_path, seed=42, val_fraction=0.15, test_fraction=0.15):
    records = load_sysu_records(csv_path)
    by_id, hashes, excluded = inspect_records(records, images)
    groups = {}
    with Path(groups_path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"sample_id", "patient_id"} <= set(reader.fieldnames or []):
            raise ValueError("Reviewed groups CSV needs sample_id,patient_id columns")
        for row in reader:
            sample_id, patient_id = row["sample_id"].strip(), row["patient_id"].strip()
            if not patient_id or sample_id not in by_id or sample_id in groups:
                raise ValueError("Groups CSV has an empty patient_id, unknown sample_id, or duplicate sample_id")
            groups[sample_id] = patient_id
    if set(groups) != set(by_id):
        raise ValueError("Groups CSV must map every SYSU sample exactly once; never infer patients from sample_id")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Use positive val/test fractions whose sum is less than one")

    # Assign even no-input rows, so they remain in each split's coverage denominator.
    included = records
    # Shared photos constrain splits, but do not establish that patient identities are equal.
    parent = {groups[record.sample_id]: groups[record.sample_id] for record in included}

    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    owner = {}
    for record in included:
        digest = hashes[record.sample_id]
        group = groups[record.sample_id]
        if digest is not None:
            if digest in owner:
                parent[find(group)] = find(owner[digest])
            else:
                owner[digest] = group
    components = sorted({find(groups[record.sample_id]) for record in included})
    if len(components) < 3:
        raise ValueError("Need at least three independent patient/photo components for train, val and test")
    random.Random(seed).shuffle(components)
    val_count = min(max(1, round(len(components) * val_fraction)), len(components) - 2)
    test_count = min(max(1, round(len(components) * test_fraction)), len(components) - val_count - 1)
    assignments = {group: "val" for group in components[:val_count]}
    assignments.update({group: "test" for group in components[val_count:val_count + test_count]})
    assignments.update({group: "train" for group in components[val_count + test_count:]})
    entries = [{"sample_id": record.sample_id, "patient_id": groups[record.sample_id],
                "label": record.label, "image_sha256": hashes[record.sample_id],
                "split": assignments[find(groups[record.sample_id])]} for record in included]
    manifest = {
        "schema_version": 1, "protocol": PROTOCOL,
        "patient_mapping_source": "externally_reviewed_anonymous_groups_csv",
        "csv_sha256": file_sha256(csv_path), "groups_sha256": file_sha256(groups_path),
        "seed": seed, "val_fraction": val_fraction, "test_fraction": test_fraction,
        "split_unit": "patient_or_shared_image_component", "label_mapping": LABELS,
        "records": entries, "excluded": excluded,
        "split_ids": {split: [entry["sample_id"] for entry in entries if entry["split"] == split] for split in SPLITS},
    }
    validate_manifest(manifest, csv_path, images, records)
    return manifest


def validate_manifest(manifest, csv_path, images, records=None):
    if manifest.get("protocol") != PROTOCOL or manifest.get("schema_version") != 1:
        raise ValueError("Use a sysu-patient-hash-v1 manifest built from reviewed anonymous patient groups")
    if manifest.get("csv_sha256") != file_sha256(csv_path) or manifest.get("label_mapping") != LABELS:
        raise ValueError("Dataset CSV or label mapping differs from manifest")
    if manifest.get("patient_mapping_source") != "externally_reviewed_anonymous_groups_csv":
        raise ValueError("Manifest requires external reviewed patient mapping provenance")
    records = load_sysu_records(csv_path) if records is None else records
    by_id, hashes, excluded = inspect_records(records, images)
    actual_excluded = {entry["sample_id"]: entry["reason"] for entry in excluded}
    recorded_excluded = {entry["sample_id"]: entry["reason"] for entry in manifest["excluded"]}
    if actual_excluded != recorded_excluded or len(recorded_excluded) != len(manifest["excluded"]):
        raise ValueError("No-input exclusion reasons differ from current dataset")
    seen, group_split, image_split = set(), {}, {}
    for entry in manifest["records"]:
        sample_id, group, split = entry["sample_id"], entry["patient_id"], entry["split"]
        if sample_id in seen or sample_id not in by_id:
            raise ValueError("Manifest sample assignments are duplicate or unknown")
        if not group or split not in SPLITS:
            raise ValueError("Every included sample needs an explicit anonymous patient_id and split")
        if by_id[sample_id].label != entry["label"] or hashes[sample_id] != entry["image_sha256"]:
            raise ValueError(f"Label or image bytes changed for sample_id={sample_id}")
        if group in group_split and group_split[group] != split:
            raise ValueError("Patient is assigned across splits")
        group_split[group] = split
        digest = hashes[sample_id]
        if digest is not None and digest in image_split and image_split[digest] != split:
            raise ValueError("Identical image occurs across splits")
        if digest is not None:
            image_split[digest] = split
        seen.add(sample_id)
    if seen != set(by_id):
        raise ValueError("Manifest must account for every CSV row")
    for split in SPLITS:
        ids = [entry["sample_id"] for entry in manifest["records"] if entry["split"] == split]
        if not ids or manifest["split_ids"].get(split) != ids:
            raise ValueError(f"Manifest {split} IDs are empty or inconsistent")
    return records


def load_manifest(path, csv_path, images):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return manifest, validate_manifest(manifest, csv_path, images)


def select_records(records, manifest, modalities=MODALITIES):
    by_id = {record.sample_id: record for record in records}
    split_by_id = {entry["sample_id"]: entry["split"] for entry in manifest["records"]}
    excluded = [{**entry, "split": split_by_id[entry["sample_id"]]} for entry in manifest["excluded"]]
    no_input_ids = {entry["sample_id"] for entry in excluded}
    splits = {}
    for split in SPLITS:
        splits[split] = []
        for sample_id in manifest["split_ids"][split]:
            if sample_id in no_input_ids:
                continue
            record = by_id[sample_id]
            present = available_modalities(record)
            if not any(present[name] for name in modalities):
                excluded.append({"sample_id": sample_id, "split": split, "reason": "all_selected_modalities_missing"})
            else:
                splits[split].append(record)
    return splits, excluded


def split_coverage(manifest, selected):
    result = {}
    for split in SPLITS:
        assigned = [entry for entry in manifest["records"] if entry["split"] == split]
        eligible = selected[split]
        eligible_ids = {record.sample_id for record in eligible}
        assigned_patients = {entry["patient_id"] for entry in assigned}
        eligible_patients = {entry["patient_id"] for entry in assigned if entry["sample_id"] in eligible_ids}
        result[split] = {
            "assigned_count": len(assigned), "eligible_count": len(eligible),
            "excluded_count": len(assigned) - len(eligible), "coverage": len(eligible) / len(assigned),
            "assigned_class_support": [sum(entry["label"] == label for entry in assigned) for label in range(3)],
            "eligible_class_support": [sum(record.label == label for record in eligible) for label in range(3)],
            "assigned_patient_count": len(assigned_patients),
            "eligible_patient_count": len(eligible_patients),
            "fully_abstained_patient_count": len(assigned_patients - eligible_patients),
        }
    return result


def manifest_summary(manifest):
    return {split: {"record_count": len(manifest["split_ids"][split]),
                    "patient_count": len({entry["patient_id"] for entry in manifest["records"] if entry["split"] == split}),
                    "class_counts": dict(Counter(entry["label"] for entry in manifest["records"] if entry["split"] == split))}
            for split in SPLITS}
