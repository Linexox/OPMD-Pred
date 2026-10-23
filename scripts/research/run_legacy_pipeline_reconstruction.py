"""Supplementary five-modality reconstruction of the old pipeline on current folds."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time

import torch

from opmd_pred.model.fusion import NumericEncoder

SPEC = importlib.util.spec_from_file_location("continuation", Path(__file__).with_name("continue_september_comparison.py"))
CONTINUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTINUE)
STUDY = CONTINUE.STUDY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=7200)
    cli = parser.parse_args()
    STUDY.DEADLINE = time.monotonic() + cli.seconds
    args, _ = CONTINUE.args_from_receipt(cli.receipt)
    torch.set_num_threads(2)
    STUDY.check_gpu()
    protocol = json.loads((args.input / "development_nested_cv_v3.json").read_text())
    original_model = STUDY.make_model
    original_pack = STUDY.pack

    def model(config, fit):
        result = original_model(config, fit)
        result.fusion.dna_encoder = NumericEncoder(1, config["dimension"]).cuda()
        return result

    def pack(data, ix, fit, training, config):
        result = original_pack(data, ix, fit, training, config)
        result["dna"] = result["dna"][:, :1]
        result["dna_mask"] = result["dna_mask"][:, :1]
        return result

    STUDY.make_model, STUDY.pack = model, pack
    initial = STUDY.configuration(encoder="vit", head="ordinal", balanced=False, dimension=256,
               dropout="legacy_subset", pretrain_epochs=30, dna_input_size=1,
               legacy_reconstruction_source_sha256=STUDY.digest(__file__),
               extension_after_first_results=True,
               differences_from_historical_run="no age/sex; corrected data and grouped nested folds; train-only TCT; four cached train views; 50epochs/patience5")
    STUDY.write(args.output / "legacy-reconstruction-protocol.json", dict(config=initial,
        purpose="reconstruct old pipeline bundle on current five-modality task, not exact reproduction or single-factor attribution",
        timing="specified after first component comparisons; exploratory supplementary analysis",
        old_SYSU_fusion_weights_reused=False, old_test_new_inference=False))
    results = []
    current = json.loads((args.output / "summary-core.json").read_text())["dino_softmax_balanced"]
    for seed in (42, 43, 44):
        config = dict(initial, seed_offset=seed - 42)
        result = STUDY.train_variant(args, protocol, f"legacy_pipeline_seed{seed}", config)
        results.append(result["metrics"])
        STUDY.write(args.output / "legacy-reconstruction-progress.json", results)
        if seed == 42 and not CONTINUE.qualifies(results[0], current):
            break
    keys = ["correct_count", "accuracy_all_assigned", "severe_to_mild_count", "two_level_error_count", "severe_recall_all_assigned", "mae"]
    STUDY.write(args.output / "legacy-reconstruction-results.json", dict(per_initialization=results,
                means={key: statistics.mean(m[key] for m in results) for key in keys},
                number_of_initializations=len(results), conditional_extension=len(results) > 1,
                old_test_new_inference=False, historical_exact_reproduction=False))


if __name__ == "__main__":
    main()
