"""Replay the saved September model, then rerun its original training once.

Run with PYTHONPATH pointing to the isolated source snapshot, not current src.
Per-record outputs and input manifests belong in a private output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
import torchvision
from torch.utils.data import DataLoader

from opmd_pred.data.sysu import SysuDataset, load_sysu_records, split_sysu_records
from opmd_pred.model.fusion import MultimodalOrdinalModel
from opmd_pred.scripts.train_multimodal import metrics, move_batch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(checkpoint, records, image_root, device):
    model = MultimodalOrdinalModel(
        dimension=checkpoint["config"]["dimension"],
        pretrained_image=False,
        tct_categories=max(record.tct_id for record in records),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    by_id = {record.sample_id: record for record in records}
    ids = checkpoint["validation_sample_ids"]
    validation = [by_id[sample_id] for sample_id in ids]
    assert len(ids) == len(set(ids)) == 21
    loader = DataLoader(SysuDataset(validation, image_root, train=False), batch_size=16, shuffle=False)
    predictions, labels, rows = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits = model(move_batch(batch, device))
            prediction = (logits.sigmoid() > 0.5).sum(dim=1).cpu()
            cumulative = logits.sigmoid().cpu()
            probabilities = torch.stack((1 - cumulative[:, 0], cumulative[:, 0] - cumulative[:, 1], cumulative[:, 1]), dim=1)
            predictions.append(prediction)
            labels.append(batch["label"])
            for i, sample_id in enumerate(batch["sample_id"]):
                rows.append(dict(sample_id=sample_id, true_label=int(batch["label"][i]),
                    predicted_label=int(prediction[i]), logits=logits[i].cpu().tolist(),
                    probabilities=probabilities[i].tolist()))
    result = metrics(torch.cat(predictions), torch.cat(labels))
    result.update(total_count=len(rows), correct_count=sum(r["true_label"] == r["predicted_label"] for r in rows),
        severe_to_mild_count=sum(r["true_label"] == 2 and r["predicted_label"] == 0 for r in rows),
        severe_recall=result["confusion_matrix"][2][2] / sum(result["confusion_matrix"][2]))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive-predictions", required=True, type=Path)
    args = parser.parse_args()
    project, source, out = args.project.resolve(), args.source.resolve(), args.output.resolve()
    if out.exists():
        raise RuntimeError("Refusing to overwrite an existing reproduction directory")
    out.mkdir(parents=True)
    os.chdir(project)
    import opmd_pred.data.sysu as sysu_module
    import opmd_pred.model.fusion as fusion_module
    assert Path(sysu_module.__file__).resolve().is_relative_to(source)
    assert Path(fusion_module.__file__).resolve().is_relative_to(source)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for the authorized training run")
    device = torch.device("cuda")
    old_path = project / "outputs/experiments/sysu/tune_subset_lr1e4/checkpoint.pt"
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    expected_config = dict(epochs=100, batch_size=16, num_workers=8, lr=1e-4,
        dimension=256, modality_dropout=0.15, dropout_mode="subset",
        unfreeze_image_blocks=0, patience=10, seed=42)
    for name, value in expected_config.items():
        assert old["config"][name] == value, (name, old["config"][name], value)
    assert old["modality_names"] == ["image", "demo", "history", "tct", "dna", "methylation"]
    csv_path = project / old["config"]["csv"]
    image_root = project / old["config"]["images"]
    fusion_path = project / old["config"]["fusion_checkpoint"]
    input_paths = [csv_path, old_path, fusion_path, args.archive_predictions.resolve()]
    records = load_sysu_records(csv_path)
    train, validation = split_sysu_records(records, seed=42)
    assert len(records) == 139 and len(train) == 118 and len(validation) == 21
    assert [record.sample_id for record in validation] == old["validation_sample_ids"]
    image_paths = sorted({image_root / r.image_path for r in records if r.image_path})
    input_paths.extend(image_paths)
    original_sha = {str(p): digest(p) for p in input_paths}
    write(out / "input-manifest.private.json", original_sha)
    write(out / "split.private.json", dict(train=[r.sample_id for r in train], validation=old["validation_sample_ids"]))
    write(out / "source-manifest.json", {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*.py"))})
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        sysu_source=str(sysu_module.__file__), fusion_source=str(fusion_module.__file__),
        source_commit="8518b14c513daae45235e29c8b5c2ad6418ef325",
        historical_runtime_versions_unknown=True)
    write(out / "environment.json", environment)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (out / "dependencies.txt").write_text(freeze.stdout, encoding="utf-8")
    replay_metrics, replay_rows = evaluate(old, records, image_root, device)
    with args.archive_predictions.open(encoding="utf-8-sig", newline="") as stream:
        archive = {r["sample_id"]: r for r in csv.DictReader(stream)}
    assert len(archive) == 21 and set(archive) == set(old["validation_sample_ids"])
    mismatches = []
    for row in replay_rows:
        ref = archive[row["sample_id"]]
        if int(ref["true_label"]) != row["true_label"] or int(ref["predicted_label"]) != row["predicted_label"]:
            mismatches.append(row["sample_id"])
    write(out / "saved-model-predictions.private.json", replay_rows)
    write(out / "saved-model-metrics.json", replay_metrics)
    replay_passed = (not mismatches and replay_metrics["correct_count"] == 15
        and replay_metrics["confusion_matrix"] == [[10, 0, 0], [4, 1, 0], [2, 0, 4]]
        and abs(replay_metrics["mae"] - 8 / 21) < 1e-6)
    write(out / "replay-check.json", dict(passed=replay_passed,
        matching_predictions=21 - len(mismatches), archive_comparison="per-record labels and predictions; archived class probabilities unavailable"))
    print(json.dumps(dict(stage="saved_model_replay", passed=replay_passed, metrics=replay_metrics)), flush=True)
    if not replay_passed:
        write(out / "mismatches.private.json", mismatches)
        raise RuntimeError("Historical replay disagrees with archive; training was not started")

    training_out = out / "retraining"
    command = [sys.executable, "-m", "opmd_pred.scripts.train_multimodal",
        "--csv", str(csv_path), "--images", str(image_root),
        "--fusion-checkpoint", str(fusion_path), "--output", str(training_out)]
    for name, value in expected_config.items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    write(out / "training-command.json", dict(argv=command, cwd=str(project),
        PYTHONPATH=os.environ.get("PYTHONPATH"),
        scope="one seed42 rerun of original fine-tuning from saved contrastive checkpoint; not rerunning Zenodo or contrastive pretraining"))
    started = time.time()
    write(out / "progress.json", dict(stage="training", started_unix=started))
    print("Starting exactly one historical fine-tuning run", flush=True)
    with (out / "retraining-console.log").open("w", encoding="utf-8") as stream:
        run = subprocess.run(command, cwd=project, stdout=stream, stderr=subprocess.STDOUT)
    if run.returncode:
        write(out / "progress.json", dict(stage="training_failed", returncode=run.returncode))
        raise RuntimeError("Original training script failed; inspect retained console")
    trained = torch.load(training_out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert trained["validation_sample_ids"] == old["validation_sample_ids"]
    training_metrics, training_rows = evaluate(trained, records, image_root, device)
    write(out / "retrained-model-predictions.private.json", training_rows)
    write(out / "retrained-model-metrics.json", training_metrics)
    unchanged = all(digest(p) == sha for p, sha in original_sha.items())
    assert unchanged, "An original input changed during reproduction"
    result = dict(saved_model_replay_passed=True, saved_model=replay_metrics,
        retrained_model=training_metrics, retrained_best_epoch=trained["epoch"],
        retrained_at_least_historical_accuracy=training_metrics["correct_count"] >= 15,
        retrained_exact_historical_predictions=all(a["predicted_label"] == b["predicted_label"] for a, b in zip(replay_rows, training_rows)),
        training_elapsed_seconds=time.time() - started, original_inputs_unchanged=unchanged,
        retrained_checkpoint_sha256=digest(training_out / "checkpoint.pt"),
        source_commit=environment["source_commit"], number_of_training_runs=1,
        historical_runtime_versions_unknown=True, validation_is_not_new_independent_test=True)
    write(out / "result-summary.json", result)
    write(out / "progress.json", dict(stage="completed", finished_unix=time.time()))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
