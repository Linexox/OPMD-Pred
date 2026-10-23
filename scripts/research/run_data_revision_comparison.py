"""Diagnostic old-field/image values on the current locked development protocol."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from opmd_pred.data.sysu import _value_mask, load_sysu_records

SPEC = importlib.util.spec_from_file_location("september_study", Path(__file__).with_name("run_september_comparison.py"))
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "cache", "reference", "old-checkpoint", "old-contrastive", "old-zenodo", "encoder", "output", "old-csv", "old-images"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=7200.)
    args = parser.parse_args()
    STUDY.DEADLINE = time.monotonic() + args.seconds
    STUDY.require(args.output.resolve().is_relative_to(Path("/home/zjkj/24334067")), "Output outside personal workspace")
    STUDY.check_gpu()
    torch.set_num_threads(2)
    protocol = json.loads((args.input / "development_nested_cv_v3.json").read_text())
    STUDY.require(STUDY.digest(args.input / "sysu_samples.csv") == protocol["csv_sha256"], "Current data changed")
    old = {r.sample_id: r for r in load_sysu_records(args.old_csv)}
    normal_factory = STUDY.make_data

    def altered_data(args, protocol, config):
        data = normal_factory(args, protocol, config)
        with np.load(args.cache / "baseline.npz", allow_pickle=False) as cached:
            new_features = cached["features"].copy()
        metadata = json.loads((args.cache / "metadata.json").read_text())
        by_sha = {metadata["source_image_sha_by_id"][sid]: new_features[i]
                  for i, sid in enumerate(data.ids) if metadata["source_image_sha_by_id"][sid]}
        source_hash = {}
        unknown_threshold = 0
        for i, sid in enumerate(data.ids):
            record = old[sid]
            STUDY.require(record.label == data.records[sid].label, "Label changed between data versions")
            if record.image_path:
                image_path = args.old_images / record.image_path
                STUDY.require(image_path.is_file(), "Old image is unavailable")
                sha = STUDY.digest(image_path)
                STUDY.require(sha in by_sha, "Old image absent from verified cache; need an explicit encoder pass")
                data.images["baseline"][i] = torch.from_numpy(by_sha[sha].copy())
                data.present[i] = True
                source_hash[sid] = sha
            else:
                data.images["baseline"][i].zero_()
                data.present[i] = False
                source_hash[sid] = None
            history, mask = _value_mask([record.smoking, record.smoking_years, record.smoking_amount,
                record.alcohol, record.alcohol_years, record.alcohol_amount,
                record.betel, record.betel_years, record.betel_amount], [1, 50, 100, 1, 50, 1000, 1, 20, 50])
            data.base["history"][i], data.base["history_mask"][i] = history, mask
            threshold = float(data.base["dna"][i, 1]) if bool(data.base["dna_mask"][i, 1]) else None
            if record.dna is None:
                threshold = None
            elif threshold is None:
                unknown_threshold += 1
            dna, mask = _value_mask([record.dna, threshold], [1, 1])
            data.base["dna"][i], data.base["dna_mask"][i] = dna, mask
            methyl, mask = _value_mask([record.methylation], [10])
            data.base["methylation"][i], data.base["methylation_mask"][i] = methyl, mask
            data.records[sid] = record
        data.eligible = data.present | data.base["history_mask"].any(1) | data.base["dna_mask"].any(1) | data.base["methylation_mask"].any(1)
        data.eligible |= torch.tensor([bool(data.records[s].tct) for s in data.ids])
        for fold in protocol["folds"]:
            for left, right in (("outer_train", "outer_holdout"), ("inner_train", "inner_val")):
                l = {source_hash[s] for s in fold[left]["sample_ids"] if source_hash[s]}
                r = {source_hash[s] for s in fold[right]["sample_ids"] if source_hash[s]}
                STUDY.require(not l.intersection(r), "Old-image version crosses frozen folds; do not silently resplit")
        STUDY.write(args.output / "old-data-diagnostic-receipt.json", dict(
            old_csv_sha256=STUDY.digest(args.old_csv), current_csv_sha256=STUDY.digest(args.input / "sysu_samples.csv"),
            old_image_sha_by_id=source_hash, records=len(data.ids), eligible=int(data.eligible.sum()),
            age_sex_used=False, labels_unchanged=True, old_test_new_inference=False,
            input_definition="old values and image associations; retain known current report threshold; otherwise threshold mask missing",
            observed_old_DNA_with_unknown_current_threshold=unknown_threshold,
            scope="diagnostic old data adaptation, not byte-exact reproduction of the historical six-modality dataset"))
        return data

    STUDY.make_data = altered_data
    config = STUDY.configuration(data_version="historical_fields_images_without_demographics",
                                old_csv_sha256=STUDY.digest(args.old_csv), diagnostic_source_sha256=STUDY.digest(__file__))
    STUDY.train_variant(args, protocol, "old_data_adaptation", config)
    STUDY.write(args.output / "completed-data-diagnostic.json", dict(completed=True, old_test_new_inference=False))


if __name__ == "__main__":
    main()
