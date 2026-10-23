"""Describe moderate-grade errors and data availability without modifying or filtering records."""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/sysu/rebuilt-20261009"
OLD = ROOT / "outputs/experiments/sysu/contrastive-ordinal-20261016/attempt1/evidence/run"
NEW = ROOT / "outputs/experiments/sysu/fused-pretraining-20261016/attempt1/evidence/run"
PRIVATE = ROOT / "outputs/experiments/sysu/moderate-quality-20261016"
PUBLIC = ROOT / "docs/daily/2026-10-16/moderate-quality-analysis.json"
LABELS = {"level1_mild_or_normal": 0, "level2_moderate": 1, "level3_severe": 2}
HISTORY = ("是否吸烟", "吸烟年限（年）", "吸烟量（支/天）", "是否饮酒", "饮酒年限(年)", "饮酒量(ml/天)", "是否嚼槟榔", "嚼槟榔年限(年)", "嚼槟榔量（颗/天）")
MISSING = {"", "NAN", "nan", "NA", "N/A", "null", "NULL", "/", "#VALUE!"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def present(value):
    return value.strip() not in MISSING


def predict(row):
    p = row["probabilities"]
    return None if p is None else int(p[1] + p[2] > .5) + int(p[2] > .5)


def main():
    protocol = read(INPUT / "development_nested_cv_v3.json")
    assert sha(INPUT / "sysu_samples.csv") == protocol["csv_sha256"]
    ids = [entry["sample_id"] for entry in protocol["development_entries"]]
    locked = set(protocol["locked_previous_test_ids"] + protocol["additional_related_locked_ids"])
    assert len(ids) == 117 and not set(ids) & locked
    with (INPUT / "sysu_samples.csv").open(encoding="utf-8-sig", newline="") as stream:
        raw = {row["sample_id"]: row for row in csv.DictReader(stream) if row["sample_id"] in ids}
    entries = {entry["sample_id"]: entry for entry in protocol["development_entries"]}
    masks, images = {}, {}
    for sid in ids:
        row = raw[sid]
        masks[sid] = [bool(entries[sid]["image_sha256"]), any(present(row[key]) for key in HISTORY),
                      present(row["TCT意见"]), present(row["DNA倍体分析结果"]), present(row["甲基化总数"])]
        if masks[sid][0]:
            path = (INPUT / row["image_path"]).resolve()
            assert path.is_relative_to(INPUT.resolve()) and sha(path) == entries[sid]["image_sha256"]
            with Image.open(path) as image:
                image.load()
                images[sid] = dict(width=image.width, height=image.height, decoded=True)
    availability = []
    for y in range(3):
        group = [sid for sid in ids if LABELS[raw[sid]["label"]] == y]
        photographed = [sid for sid in group if sid in images]
        availability.append(dict(label=y, records=len(group), modalities=[sum(masks[sid][j] for sid in group) for j in range(5)],
            all_five=sum(all(masks[sid]) for sid in group), no_input=sum(not any(masks[sid]) for sid in group),
            count_by_modalities=[sum(sum(masks[sid]) == k for sid in group) for k in range(6)],
            image_decode_failures=0, image_minimum_side_range=[min(min(images[sid]["width"], images[sid]["height"]) for sid in photographed),
                max(min(images[sid]["width"], images[sid]["height"]) for sid in photographed)],
            tct_value_counts=dict(Counter(raw[sid]["TCT意见"] for sid in group if masks[sid][2]))))
    methods = [(OLD, mode) for mode in ("balanced_reference", "softmax", "ordinal", "softmax_pretrain", "ordinal_pretrain")]
    methods += [(NEW, "fused_full_class"), (NEW, "fused_subset_class")]
    results, private, fingerprints = {}, {}, {}
    for parent, method in methods:
        per_seed, detailed = [], []
        for seed in (42, 43, 44):
            path = parent / "runs" / f"{method}_seed{seed}" / "oof_results.json"
            fingerprints[str(path.relative_to(ROOT))] = sha(path)
            result = read(path)
            rows = {row["sample_id"]: row for row in result["records"]}
            assert set(rows) == set(ids)
            assert all(row["label"] == LABELS[raw[sid]["label"]] for sid, row in rows.items())
            intermediate = [sid for sid in ids if rows[sid]["label"] == 1]
            predicted = [predict(rows[sid]) for sid in intermediate]
            groups = dict(all=intermediate, image_present=[sid for sid in intermediate if masks[sid][0]],
                no_image=[sid for sid in intermediate if not masks[sid][0]],
                all_five=[sid for sid in intermediate if all(masks[sid])],
                fewer_than_five=[sid for sid in intermediate if not all(masks[sid])])
            strata = {name: dict(records=len(group), correct=sum(predict(rows[sid]) == 1 for sid in group),
                         recall=sum(predict(rows[sid]) == 1 for sid in group) / len(group) if group else None) for name, group in groups.items()}
            per_seed.append(dict(seed=seed, accuracy=result["metrics"]["accuracy_all_assigned"], moderate_records=len(intermediate),
                predicted_low=predicted.count(0), predicted_moderate=predicted.count(1), predicted_severe=predicted.count(2),
                abstained=predicted.count(None), strata=strata,
                total_predicted_moderate=sum(predict(row) == 1 for row in rows.values())))
            detailed.append(dict(seed=seed, records=[dict(sample_id=sid, prediction=predict(rows[sid]), available=masks[sid]) for sid in intermediate]))
        gaps = []
        if result["config"]["head"] == "ordinal":
            for seed in (42, 43, 44):
                for fold in range(5):
                    checkpoint = parent / "runs" / f"{method}_seed{seed}" / f"fold{fold}/refit/classifier.pt"
                    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    gap = float(torch.nn.functional.softplus(state["state"]["ordinal_head.threshold_gap"]))
                    gaps.append(dict(seed=seed, fold=fold, gap=gap, maximum_possible_middle_probability=math.tanh(gap / 4),
                                     selected_epoch=state["selected_epoch"]))
        means = {key: statistics.mean(item[key] for item in per_seed) for key in
                 ("accuracy", "predicted_low", "predicted_moderate", "predicted_severe", "abstained", "total_predicted_moderate")}
        means["moderate_recall"] = means["predicted_moderate"] / 31
        means["strata"] = {name: dict(records=per_seed[0]["strata"][name]["records"],
            recall=statistics.mean(item["strata"][name]["recall"] for item in per_seed)) for name in per_seed[0]["strata"]}
        results[method] = dict(per_seed=per_seed, means=means,
            threshold_gap_range=[min(x["gap"] for x in gaps), max(x["gap"] for x in gaps)] if gaps else None,
            maximum_middle_probability_range=[min(x["maximum_possible_middle_probability"] for x in gaps),
                max(x["maximum_possible_middle_probability"] for x in gaps)] if gaps else None)
        private[method] = dict(moderate_records=detailed, checkpoints=gaps)
    public = dict(records=117, class_availability=availability, methods=results,
        image_clinical_quality_review_performed=False, histology_regrading_performed=False,
        original_data_modified=False, records_excluded=False, training_performed=False,
        locked_test_evaluated=False, probability_decision="All methods retain archived fixed cumulative median thresholds 0.5, not argmax",
        limitations=["Strata are descriptive and small; availability is not clinical quality", "Middle probability ceiling is an architectural diagnostic; decisions use cumulative thresholds, so middle predictions remain possible"],
        source_sha256=sha(Path(__file__)), csv_sha256=sha(INPUT / "sysu_samples.csv"), protocol_sha256=sha(INPUT / "development_nested_cv_v3.json"))
    write(PUBLIC, public)
    write(PRIVATE / "case-review.private.json", private)
    write(PRIVATE / "inputs-sha.json", fingerprints)
    print(json.dumps(dict(availability=availability,
        methods={name: dict(means=item["means"], gap=item["threshold_gap_range"], middle_ceiling=item["maximum_middle_probability_range"]) for name, item in results.items()}), ensure_ascii=True))


if __name__ == "__main__":
    main()
