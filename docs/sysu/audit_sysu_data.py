"""Read-only SYSU audit. Outputs aggregates only; never outputs row identifiers."""

import csv
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from opmd_pred.data.sysu import load_sysu_records, split_sysu_records

MISSING = {"", "NAN", "nan", "NA", "N/A", "null", "NULL", "/", "#VALUE!"}
SOURCES = [("level1", "level1_mild_or_normal", "mild+normal.xlsx"),
           ("level2", "level2_moderate", "moderate.xlsx"),
           ("level3", "level3_severe", "severe.xlsx")]
csv_path = ROOT / "data/SYSU_processed/sysu_samples.csv"
rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
labels = [label for _, label, _ in SOURCES]
groups = {label: [r for r in rows if r["label"] == label] for label in labels}
present = lambda value: str("" if value is None else value).strip() not in MISSING
history_fields = [k for k in rows[0] if k.startswith(("是否", "吸烟", "饮酒", "嚼槟榔"))]

def modalities(row):
    return {"image": present(row["image_path"]),
            "history": any(present(row[k]) for k in history_fields),
            "tct": present(row["TCT意见"]),
            "dna": present(row["DNA倍体分析结果"]),
            "methylation": present(row["甲基化总数"])}

def patterns(selection, names):
    counts = Counter("".join("1" if modalities(r)[n] else "0" for n in names) for r in selection)
    return {"bit_order": names, "counts": dict(sorted(counts.items()))}

def duplicate_summary(buckets):
    repeats = [v for v in buckets.values() if len(v) > 1]
    return {"groups": len(repeats), "records": sum(map(len, repeats)),
            "extra_records": sum(len(v) - 1 for v in repeats),
            "cross_label_groups": sum(len({r["label"] for r in v}) > 1 for v in repeats)}

field_counts = {k: {"observed": sum(present(r[k]) for r in rows),
                    "missing": sum(not present(r[k]) for r in rows)}
                for k in rows[0] if k not in ("sample_id", "label")}
images = {"declared": 0, "exists": 0, "readable": 0, "unreadable": 0, "missing_files": 0}
sha_groups, pixel_groups, image_sha_by_id = defaultdict(list), defaultdict(list), {}
sizes = Counter()
for row in rows:
    if not row["image_path"]:
        continue
    images["declared"] += 1
    path = csv_path.parent / row["image_path"]
    if not path.is_file():
        images["missing_files"] += 1
        continue
    images["exists"] += 1
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    sha_groups[sha].append(row)
    image_sha_by_id[row["sample_id"]] = sha
    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            rgb.load()
            size = rgb.size
            pixel_sha = hashlib.sha256(str(size).encode() + rgb.tobytes()).hexdigest()
        images["readable"] += 1
        pixel_groups[pixel_sha].append(row)
        sizes[str(size)] += 1
    except Exception:
        images["unreadable"] += 1
images["byte_duplicates"] = duplicate_summary(sha_groups)
images["decoded_rgb_duplicates"] = duplicate_summary(pixel_groups)
images["dimensions"] = dict(sizes)
images["duplicate_label_combinations"] = dict(Counter(" + ".join(sorted({r["label"] for r in group})) for group in sha_groups.values() if len(group) > 1))

# Only sample/source references; keep outside shared docs. No personal names.
private_dir = ROOT / "tmp/sysu-private"
private_dir.mkdir(parents=True, exist_ok=True)
with (private_dir / "duplicate-image-review-2026-10-08.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["duplicate_group", "sample_id", "label", "source_workbook", "source_sheet", "source_excel_row", "image_sha256"])
    writer.writeheader()
    duplicate_index = 0
    for sha, group in sha_groups.items():
        if len(group) < 2:
            continue
        duplicate_index += 1
        for row in group:
            level, excel_row = row["sample_id"].split("_")
            source = next(s for s in SOURCES if s[0] == level)
            writer.writerow({"duplicate_group": duplicate_index, "sample_id": row["sample_id"], "label": row["label"],
                             "source_workbook": f"data/sysu/{level}/{source[2]}",
                             "source_sheet": {"level1": "mild", "level2": "moderate", "level3": "severe"}[level],
                             "source_excel_row": int(excel_row), "image_sha256": sha})

name_groups = defaultdict(list)
name_by_id = {}
raw = {}
GENES = ["CDKN2A", "TIMP3", "ZNF582", "TMEFF2", "TFPI2", "SHOX2", "PAX1", "MGMT", "SOX1", "DAPK1"]
normalize = lambda s: str(s or "").replace(" ", "").replace("　", "").strip()
def directory_name(s):
    s = normalize(s)
    for prefix in ("new_", "v", "删除_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s

for level, label, filename in SOURCES:
    directory = ROOT / "data/sysu" / level
    workbook = load_workbook(directory / filename, read_only=True, data_only=True)
    active_sheet = workbook.active.title
    expected_sheet = {"level1": "mild", "level2": "moderate", "level3": "severe"}[level]
    values = list(workbook[expected_sheet].iter_rows(values_only=True))
    id_headers = {"patient_id", "encounter_id", "visit_id", "病历号", "患者编号", "病人编号", "住院号", "门诊号", "样本编号", "就诊编号", "病理编号"}
    sheet_inventory = []
    for ws in workbook.worksheets:
        first = [str(v).strip() if v is not None else "" for v in next(ws.iter_rows(values_only=True))]
        sheet_inventory.append({"name": ws.title, "rows": ws.max_row, "columns": ws.max_column,
                                "known_training_headers": [h for h in first if h in rows[0]],
                                "candidate_identity_headers": [h for h in first if h.lower() in id_headers],
                                "date_headers": [h for h in first if "日期" in h or h.lower() in {"date", "time", "timestamp"}]})
    headers = [str(v).strip() if v is not None else "" for v in values[0]]
    data = [dict(zip(headers, vals)) for vals in values[1:] if any(vals)]
    workbook.close()
    folder_matches = {"zero": 0, "one": 0, "multiple": 0, "prefix_only": 0,
                      "processed_image_byte_match": 0, "processed_image_byte_mismatch": 0}
    folders = [p for p in directory.iterdir() if p.is_dir()]
    for excel_row, row in enumerate(data, start=2):
        name = normalize(row.get("name"))
        sample = f"{level}_{excel_row:03d}"
        if name:
            name_groups[name].append({"label": label, "sample_id": sample})
            name_by_id[sample] = name
        matches = [p for p in folders if directory_name(p.name).startswith(name)
                   or name.startswith(directory_name(p.name))]
        folder_matches["zero" if not matches else "one" if len(matches) == 1 else "multiple"] += 1
        folder_matches["prefix_only"] += bool(matches and all(directory_name(p.name) != name for p in matches))
        if sample in image_sha_by_id:
            source_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for folder in matches
                             for p in folder.iterdir() if p.is_file() and p.stem == "病损"]
            key = "processed_image_byte_match" if image_sha_by_id[sample] in source_hashes else "processed_image_byte_mismatch"
            folder_matches[key] += 1
    formula_wb = load_workbook(directory / filename, read_only=True, data_only=False)
    formula_rows = list(formula_wb[expected_sheet].iter_rows(values_only=True))
    formula_counts = Counter()
    formula_structure = None
    for vals in formula_rows[1:]:
        for k, value in zip(headers, vals):
            if isinstance(value, str) and value.startswith("="):
                formula_counts[k] += 1
                if k == "DNA倍体分析结果" and formula_structure is None:
                    references = re.findall(r"([A-Z]+)[0-9]+", value)
                    formula_structure = {"expression": re.sub(r"[0-9]+", "ROW", value),
                                         "referenced_headers": [headers[column_index_from_string(c) - 1] for c in references]}
    formula_wb.close()
    processed_by_id = {r["sample_id"]: r for r in groups[label]}
    clean = lambda value: "" if str(value if value is not None else "").strip() in MISSING else "30" if str(value).strip() == "几十年" else str(value).strip()
    mismatch_fields = Counter()
    for excel_row, row in enumerate(data, start=2):
        processed = processed_by_id.get(f"{level}_{excel_row:03d}")
        if processed is None:
            mismatch_fields["missing_processed_record"] += 1
            continue
        for field in rows[0]:
            if field in ("sample_id", "label", "image_path"):
                continue
            if clean(row.get(field)) != processed[field]:
                mismatch_fields[field] += 1
    raw[label] = {"rows": len(data), "columns": headers, "active_sheet": active_sheet,
                  "xlsx_sha256": hashlib.sha256((directory / filename).read_bytes()).hexdigest(),
                  "audited_named_sheet": expected_sheet, "sheet_inventory": sheet_inventory,
                  "name_observed": sum(bool(normalize(r.get("name"))) for r in data),
                  "processed_comparison_field_mismatch_counts": dict(mismatch_fields),
                  "gene_observed_counts": {gene: sum(present(r.get(gene)) for r in data) for gene in GENES},
                  "gene_completeness_counts": dict(Counter(str(sum(present(r.get(g)) for g in GENES)) for r in data)),
                  "dna_formula_structure": formula_structure,
                  "folder_linkage": folder_matches, "formula_counts": dict(formula_counts),
                  "exact_stem_lesion_files": sum(p.is_file() and p.stem == "病损" for p in directory.rglob("*")),
                  "numbered_stem_lesion_files": sum(p.is_file() and p.stem.startswith("病损") and p.stem != "病损" for p in directory.rglob("*"))}

records = load_sysu_records(csv_path)
train, val = split_sysu_records(records, seed=42)
train_ids, val_ids = {r.sample_id for r in train}, {r.sample_id for r in val}
train_tct = {r.tct for r in train if r.tct}
val_tct = {r.tct for r in val if r.tct}
train_names = {name_by_id[s] for s in train_ids if s in name_by_id}
val_names = {name_by_id[s] for s in val_ids if s in name_by_id}
train_sha = {image_sha_by_id[s] for s in train_ids if s in image_sha_by_id}
val_sha = {image_sha_by_id[s] for s in val_ids if s in image_sha_by_id}
tab_fields = [k for k in rows[0] if k not in ("sample_id", "label", "image_path")]
tab_groups = defaultdict(list)
for r in rows:
    tab_groups[tuple(r[k] for k in tab_fields)].append(r)
out = {"audit_type": "aggregate_read_only_not_model_evaluation", "date": "2026-10-08",
       "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
       "record_count": len(rows), "unique_sample_ids": len({r["sample_id"] for r in rows}),
       "true_patient_count": None, "patient_id_available_in_processed": False,
       "field_counts": field_counts, "images": images,
       "numeric_modality_summary": {field: {label: {"observed": len(values), "min": min(values) if values else None,
                                                     "max": max(values) if values else None, "median": statistics.median(values) if values else None,
                                                     "zero_count": sum(v == 0 for v in values)}
                                              for label, g in groups.items()
                                              for values in [[float(r[field]) for r in g if present(r[field])]]}
                                      for field in ["DNA倍体分析结果", "甲基化总数"]},
       "aggregate_tct_categories": {category: {label: sum(r["TCT意见"] == category for r in groups[label]) for label in labels}
                                    for category in sorted({r["TCT意见"] for r in rows if r["TCT意见"]})},
       "by_class": {label: {"records": len(g), "modalities_observed": {n: sum(modalities(r)[n] for r in g) for n in modalities(rows[0])},
                             "patterns_all_non_demographic": patterns(g, list(modalities(rows[0]))),
                             "patterns_focus": patterns(g, ["image", "dna", "methylation"])} for label, g in groups.items()},
       "patterns_all_non_demographic": patterns(rows, list(modalities(rows[0]))),
       "patterns_focus": patterns(rows, ["image", "dna", "methylation"]),
       "focus_all_present": sum(all(modalities(r)[n] for n in ["image", "dna", "methylation"]) for r in rows),
       "focus_all_missing": sum(not any(modalities(r)[n] for n in ["image", "dna", "methylation"]) for r in rows),
       "raw_sources": raw, "same_normalized_name": duplicate_summary(name_groups),
       "distinct_normalized_names_not_confirmed_patients": len(name_groups),
       "exact_tabular_duplicates": duplicate_summary(tab_groups),
       "default_seed42_record_split": {"train": len(train), "validation": len(val),
           "train_class_counts": dict(Counter(r.label for r in train)), "validation_class_counts": dict(Counter(r.label for r in val)),
           "same_name_overlap_count_not_confirmed_identity": len(train_names & val_names),
           "identical_image_sha_overlap_count": len(train_sha & val_sha),
           "all_tct_vocabulary": len(train_tct | val_tct), "train_tct_vocabulary": len(train_tct),
           "validation_only_tct_categories": len(val_tct - train_tct),
           "validation_records_with_unseen_tct": sum(bool(r.tct) and r.tct not in train_tct for r in val)}}
destination = ROOT / "docs/sysu/data-audit-2026-10-08.json"
destination.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k not in ("raw_sources", "by_class", "images")}, ensure_ascii=True, indent=2))
print(json.dumps({"images": images, "by_class": {k: v["modalities_observed"] for k, v in out["by_class"].items()},
                  "folder_linkage": {k: v["folder_linkage"] for k, v in raw.items()},
                  "raw_lesion_files": {k: [v["exact_stem_lesion_files"], v["numbered_stem_lesion_files"]] for k,v in raw.items()}}, ensure_ascii=True, indent=2))
