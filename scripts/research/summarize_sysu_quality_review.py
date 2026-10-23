"""Combine completed source reading and photo review without changing training data."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/sysu/rebuilt-20261009"
OUT = ROOT / "outputs/experiments/sysu/moderate-quality-20261016/source-review"
PUBLIC = ROOT / "docs/daily/2026-10-16"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def date_value(value):
    if not value:
        return None
    return datetime.strptime(value.replace("/", "-"), "%Y-%m-%d").date()


def main():
    prepared = read(OUT / "preparation-summary.json")
    cases = read(OUT / "source-cases.private.json")
    reads = read(OUT / "histology-verified-reads.private.json")
    blind = read(OUT / "photo-review.blind.json")
    photos = {r["code"]: r for r in read(OUT / "photo-map.private.json")}
    cv = read(INPUT / "development_nested_cv_v3.json")
    entries = {r["sample_id"]: r for r in cv["development_entries"]}
    provenance = {r["sample_id"]: r for r in read(INPUT / "record_provenance.private.json")}
    with (INPUT / "sysu_samples.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = {r["sample_id"]: r for r in csv.DictReader(stream)}
    checks = []
    for item in read(OUT / "file-inventory.private.json"):
        checks.append({"path": item["path"], "unchanged": sha(item["path"]) == item["sha256"]})
    for item in photos.values():
        checks.append({"path": item["cached_image_path"], "unchanged": sha(item["cached_image_path"]) == item["sha256"]})
    for item in reads:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("A visually reviewed report changed")
    if not all(item["unchanged"] for item in checks):
        raise ValueError("A source file changed during review")
    if sha(INPUT / "sysu_samples.csv") != prepared["source_csv_sha"] or sha(INPUT / "development_nested_cv_v3.json") != prepared["split_sha"]:
        raise ValueError("Frozen data or split changed")
    grouped_reads = defaultdict(list)
    for item in reads:
        grouped_reads[item["sample_id"]].append(item)
    joined_photos = [{**photos[r["code"]], **r} for r in blind["records"]]
    by_sha = defaultdict(list)
    for item in joined_photos:
        by_sha[item["sha256"]].append(item)
    duplicates = []
    for image_sha, group in by_sha.items():
        if len(group) < 2:
            continue
        ids = [r["sample_id"] for r in group]
        membership = [{sid: [f["fold"] for f in cv["folds"] if sid in f["outer_holdout"]["sample_ids"]] for sid in ids}]
        outer_ok = all(all((sid in f["outer_holdout"]["sample_ids"]) == (ids[0] in f["outer_holdout"]["sample_ids"]) for sid in ids) for f in cv["folds"])
        inner_ok = all(all((sid in f["inner_val"]["sample_ids"]) == (ids[0] in f["inner_val"]["sample_ids"]) for sid in ids) for f in cv["folds"])
        dna_shas = [(provenance[sid].get("dna") or {}).get("report_sha256") for sid in ids]
        duplicates.append({"sample_ids": ids, "labels": [entries[sid]["label"] for sid in ids], "image_sha256": image_sha,
            "same_dna_report": bool(dna_shas[0]) and len(set(dna_shas)) == 1,
            "same_cv_component": len({entries[sid]["cv_component"] for sid in ids}) == 1,
            "outer_fold_grouped": outer_ok, "inner_fold_grouped": inner_ok, "holdout_membership": membership,
            "different_table_fields": [k for k in rows[ids[0]] if k not in {"sample_id", "label", "image_path"} and any(rows[sid][k] != rows[ids[0]][k] for sid in ids)]})
    classifications = {}
    priority = {"level2_016": "报告中度并有局部重度", "level2_018": "报告提及局部恶变而非明确中度", "level2_035": "所见报告仅轻度"}
    case_results = []
    late_collection = []
    for case in cases:
        sid = case["sample_id"]
        source_reads = grouped_reads[sid]
        diagnostic = [r for r in source_reads if r.get("reports")]
        # Historical reports and consultations remain recorded; use the explicit
        # later report only for this evidence description, never to change labels.
        selected = [r for r in diagnostic if r["code"] not in {"R004", "R006"}]
        categories = [report["category"] for r in selected for report in r["reports"]]
        if sid in priority:
            status = "priority_grade_review"
        elif not diagnostic:
            status = "no_histology_found_in_bound_folder"
        elif "mild_to_moderate" in categories:
            status = "mild_to_moderate_interval"
        elif "multiple_specimens_mild_and_moderate" in categories:
            status = "different_grades_by_specimen"
        elif any(c in {"focal_moderate_with_mild", "peripheral_moderate"} for c in categories):
            status = "focal_or_peripheral_moderate"
        else:
            status = "moderate_wording"
        classifications[sid] = status
        dna = case.get("frozen_dna") or {}
        collection = date_value((dna.get("dates") or {}).get("collection"))
        final_dates = [date_value(report.get("reported")) for r in selected for report in r["reports"] if report.get("reported")]
        after = bool(collection and final_dates and collection > max(final_dates))
        if after:
            late_collection.append(sid)
        photo = next((r for r in joined_photos if r["sample_id"] == sid), None)
        result = {**case, "audit_grade_status": status, "priority_reason": priority.get(sid),
            "pathology_source_reads": source_reads, "photo_review": photo,
            "grade_changed": False, "same_encounter_and_target_confirmed": False,
            "dna_collection_after_selected_pathology_report": after,
            "dna_collection_after_report_is_not_automatic_mismatch": True,
            "original_grade_retained": True}
        case_results.append(result)
    flags = Counter(flag for r in joined_photos for flag in r["flags"])
    summary = {
        "scope": "117 development records only; 31 moderate source audits and 75 selected photos; no locked-test evaluation",
        "reviewer": "assistant source transcription and technical photo review, not clinician adjudication",
        "moderate_records": len(cases), "histology_evidence_records": sum(bool(r["pathology_source_reads"]) for r in case_results),
        "histology_jpg_files_viewed": sum(Path(r["path"]).suffix.lower() == ".jpg" for r in reads),
        "histology_pdf_files_viewed": sum(Path(r["path"]).suffix.lower() == ".pdf" for r in reads),
        "grade_status_counts": dict(Counter(classifications.values())),
        "photo_count": len(joined_photos), "photo_native_review_count": blind["native_images_reviewed"],
        "photo_original_unnumbered_match_count": sum(bool(r["selected_unnumbered_source_matches"]) for r in joined_photos),
        "photo_flagged_count": sum(bool(r["flags"]) for r in joined_photos), "photo_flag_counts": dict(flags),
        "photo_flagged_by_grade": dict(Counter(str(r["label"]) for r in joined_photos if r["flags"])),
        "photo_major_defocus_by_grade": dict(Counter(str(r["label"]) for r in joined_photos if "major_defocus" in r["flags"])),
        "same_photo_duplicate_groups": len(duplicates),
        "same_photo_different_grade_groups": sum(len(set(r["labels"])) > 1 for r in duplicates),
        "duplicates_all_grouped_in_outer_and_inner_splits": all(r["outer_fold_grouped"] and r["inner_fold_grouped"] for r in duplicates),
        "dna_explicit_collection_after_available_pathology_report_count": len(late_collection),
        "nondefault_exif_orientation_count": sum(r["exif_orientation"] not in (None, 1) for r in joined_photos),
        "source_file_checks": len(checks), "source_files_unchanged": all(r["unchanged"] for r in checks),
        "csv_sha256": sha(INPUT / "sysu_samples.csv"), "split_sha256": sha(INPUT / "development_nested_cv_v3.json"),
        "records_deleted": 0, "labels_changed": 0, "models_trained": 0, "new_predictions": 0,
        "clinical_usability_confirmed": False,
    }
    write(OUT / "case-review.private.json", case_results)
    write(OUT / "photo-review.joined.private.json", joined_photos)
    write(OUT / "duplicate-review.private.json", duplicates)
    write(OUT / "source-unchanged-checks.private.json", checks)
    with (OUT / "case-review.private.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        names = ["记录编号", "病理依据情况", "优先核对原因", "报告诊断原文", "原报告位置", "DNA报告日期字段", "DNA采样晚于现存病理报告", "照片观察", "是否确认同次同部位", "是否修改标签"]
        status_text = {"priority_grade_review": "等级依据优先核对", "no_histology_found_in_bound_folder": "对应原始目录未找到病理报告", "mild_to_moderate_interval": "报告写轻度至中度", "different_grades_by_specimen": "同报告不同标本有轻度与中度", "focal_or_peripheral_moderate": "局部或周边中度", "moderate_wording": "报告有中度表述"}
        flag_text = {"localized_soft_focus": "局部偏软或失焦", "major_defocus": "明显失焦", "display_recapture": "屏幕翻拍", "annotation_overlay": "叠加标记", "limited_target_view_confirmation": "目标部位需核对", "visible_suture_like_material_timing_check": "可见疑似缝线，需核对拍摄时点", "possible_staining_or_application": "可见涂染或涂抹器械", "low_native_resolution": "原始尺寸小"}
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for r in case_results:
            writer.writerow({"记录编号": r["sample_id"], "病理依据情况": status_text[r["audit_grade_status"]], "优先核对原因": r["priority_reason"],
                "报告诊断原文": " | ".join(f"{s['code']}: {t['grade_text']}" for s in r["pathology_source_reads"] for t in s.get("reports", [])),
                "原报告位置": " | ".join(s["path"] for s in r["pathology_source_reads"]),
                "DNA报告日期字段": json.dumps((r.get("frozen_dna") or {}).get("dates"), ensure_ascii=False),
                "DNA采样晚于现存病理报告": "是" if r["dna_collection_after_selected_pathology_report"] else "未发现或日期不足",
                "照片观察": " | ".join(flag_text[x] for x in (r.get("photo_review") or {}).get("flags", [])),
                "是否确认同次同部位": "尚未完全确认", "是否修改标签": "否"})
    write(OUT / "summary.json", summary)
    write(PUBLIC / "sysu-quality-review.json", summary)
    files = ["histology-verified-reads.private.json", "photo-review.blind.json", "quality-rubric.md", "case-review.private.json", "photo-review.joined.private.json", "duplicate-review.private.json", "source-unchanged-checks.private.json", "summary.json", "case-review.private.csv"]
    write(OUT / "review-evidence-sha.json", {name: sha(OUT / name) for name in files} | {"summary_script_sha256": sha(Path(__file__))})
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
