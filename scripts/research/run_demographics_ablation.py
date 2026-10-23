"""Compare the frozen 5-input contrastive/ordinal method with age and sex added."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import time

import torch
from torch import nn

RUNNER_PATH = Path(__file__).with_name("run_contrastive_ordinal_cv.py")
SPEC = importlib.util.spec_from_file_location("contrastive_ordinal", RUNNER_PATH)
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
S = C.S


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path):
    return S.digest(path)


class DemographicData(S.REFERENCE.Data):
    def __init__(self, csv_path, protocol, cache, demographics_path):
        super().__init__(csv_path, protocol, cache)
        with Path(demographics_path).open(encoding="utf-8-sig", newline="") as stream:
            rows = {row["sample_id"]: row for row in csv.DictReader(stream)}
        S.require(len(rows) == len(self.ids) and set(rows) == set(self.ids),
                  "Demographic file must contain exactly the 117 development records")
        from opmd_pred.data.sysu import _value_mask

        values, masks = [], []
        for sample_id in self.ids:
            row = rows[sample_id]
            S.require(int(row["label"]) == self.records[sample_id].label,
                      f"Demographic label differs for {sample_id}")
            age = float(row["age"]) if row["age"] else None
            sex = float(row["sex"]) if row["sex"] else None
            S.require(age is None or 0 <= age <= 120, f"Invalid age for {sample_id}")
            S.require(sex is None or sex in (0.0, 1.0), f"Invalid sex code for {sample_id}")
            value, mask = _value_mask([age, sex], [100.0, 1.0])
            values.append(value)
            masks.append(mask)
        self.demographics = torch.stack(values)
        self.demographics_mask = torch.stack(masks)

    def pack(self, indices, fit, training):
        batch = super().pack(indices, fit, training)
        batch["demographics"] = self.demographics[indices].to("cuda")
        batch["demographics_mask"] = self.demographics_mask[indices].to("cuda")
        return batch


def install_demographic_model_support():
    from opmd_pred.model.fusion import NumericEncoder
    from opmd_pred.scripts.train_multimodal import apply_modality_dropout

    base_configuration = S.configuration
    base_make_model = S.make_model
    base_embedding_inputs = S.embedding_inputs

    def configuration(**changes):
        result = base_configuration(**changes)
        result["include_demographics"] = True
        return result

    def make_model(config, fit):
        model = base_make_model(config, fit)
        if config.get("include_demographics"):
            fusion = model.fusion
            fusion.demographics_encoder = NumericEncoder(2, config["dimension"]).cuda()
            fusion.names = (*fusion.names, "demographics")
            fusion.missing_tokens["demographics"] = nn.Parameter(
                torch.randn(config["dimension"], device="cuda") * 0.02
            )
        return model

    def embedding_inputs(model, batch):
        embeddings, present = base_embedding_inputs(model, batch)
        if "demographics" in model.fusion.names:
            fusion = model.fusion
            embeddings["demographics"] = fusion.demographics_encoder(
                batch["demographics"], batch["demographics_mask"]
            )
            present["demographics"] = batch["demographics_mask"].bool().any(1)
        return embeddings, present

    def dropout(batch, config):
        names = ("image", "history", "tct", "dna", "methylation", "demographics")
        return apply_modality_dropout(batch, 0.15, "bernoulli", names)

    S.configuration = configuration
    S.make_model = make_model
    S.embedding_inputs = embedding_inputs
    S.dropout = dropout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--demographics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=7200)
    args = parser.parse_args()

    S.DEADLINE = time.monotonic() + args.seconds
    torch.set_num_threads(2)
    S.check_gpu()
    S.require(args.output.resolve().is_relative_to(Path("/home/zjkj/24334067")),
              "Output outside personal directory")
    protocol = read(args.input / "development_nested_cv_v3.json")
    S.require(digest(args.input / "sysu_samples.csv") == protocol["csv_sha256"],
              "SYSU table differs from frozen split")
    rows = list(csv.DictReader(args.demographics.open(encoding="utf-8-sig", newline="")))
    S.require(len(rows) == 117 and len({row["sample_id"] for row in rows}) == 117,
              "Expected exactly 117 unique development records")
    age_count = sum(bool(row["age"]) for row in rows)
    sex_count = sum(bool(row["sex"]) for row in rows)
    S.require(age_count == 112 and sex_count == 117,
              "Demographic coverage changed from the locally audited source")
    args.output.mkdir(parents=True, exist_ok=True)
    run_config = dict(
        source_sha256=digest(Path(__file__)),
        contrastive_runner_sha256=digest(RUNNER_PATH),
        input_csv_sha256=digest(args.input / "sysu_samples.csv"),
        split_sha256=digest(args.input / "development_nested_cv_v3.json"),
        image_cache_sha256=digest(args.cache / "baseline.npz"),
        demographic_csv_sha256=digest(args.demographics),
        cohort_records=117,
        age_available=age_count,
        sex_available=sex_count,
        both_available=sum(bool(row["age"] and row["sex"]) for row in rows),
        method="30-epoch multimodal contrastive pretraining + ordinal classification",
        input_modalities=["image", "history", "TCT", "DNA", "methylation", "age+sex"],
        seeds=[42, 43, 44],
        folds=5,
        original_all_missing_records_remain_abstained=True,
        locked_test_evaluated=False,
    )
    config_path = args.output / "protocol.json"
    if config_path.exists():
        S.require(read(config_path) == run_config, "Saved run protocol differs")
    else:
        S.write(config_path, run_config)

    install_demographic_model_support()
    data = DemographicData(args.input / "sysu_samples.csv", protocol, args.cache,
                           args.demographics)
    results = []
    for seed in (42, 43, 44):
        result = C.run_fixed(args, protocol, data, "ordinal_pretrain", seed)
        result["method"] = "ordinal_pretrain_demographics"
        seed_root = args.output / "runs" / f"ordinal_pretrain_demographics_seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        S.write(seed_root / "oof_results.json", result)
        results.append(result)
        S.write(args.output / "progress.json", dict(stage="training", seeds_completed=len(results),
                                                       total_seeds=3, last_seed=seed))

    keys = ("accuracy_all_assigned", "severe_recall_all_assigned", "mae", "abstained_count")
    summary = {
        "method": "ordinal_pretrain_demographics",
        "per_seed": [dict(seed=result["seed"], metrics=result["metrics"])
                     for result in results],
        "means": {key: statistics.mean(result["metrics"][key] for result in results)
                  for key in keys},
        "cohort_records": 117,
    }
    S.write(args.output / "summary.json", summary)
    S.write(args.output / "progress.json", dict(stage="training_completed", seeds_completed=3))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
