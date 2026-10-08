"""Explain a fixed saved validation image; no training or test sample selection."""

import argparse
from pathlib import Path

import torch
from PIL import Image, __version__ as pillow_version

from opmd_pred.analysis.image_attribution import TARGETS, image_geometry, integrated_gradients, save_heatmap
from opmd_pred.data.sysu import SysuDataset, assign_tct_ids
from opmd_pred.data.sysu_protocol import file_sha256, select_records, validate_manifest
from opmd_pred.scripts.train_multimodal import move_batch
from opmd_pred.scripts.train_sysu import read_checkpoint, restore_checkpoint, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-id", required=True, help="A sample ID from checkpoint's saved validation split only")
    parser.add_argument("--output", required=True, help="New directory; never overwrite")
    parser.add_argument("--target", choices=TARGETS, default="severe")
    parser.add_argument("--baseline", choices=["mean", "black", "blur"], default="mean")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing image attribution")
    # CSV bytes, all image bytes and saved split are validated on CPU first.
    checkpoint_sha = file_sha256(args.checkpoint)
    checkpoint = read_checkpoint(args.checkpoint)
    records = validate_manifest(checkpoint["split_manifest"], args.csv, args.images)
    config = checkpoint["model_config"]
    splits, excluded = select_records(records, checkpoint["split_manifest"], config["modalities"])
    actual_ids = {name: [record.sample_id for record in selected] for name, selected in splits.items()}
    if actual_ids != checkpoint["effective_split_ids"] or excluded != checkpoint["excluded"]:
        raise ValueError("Effective splits or exclusions differ from fixed checkpoint")
    if args.sample_id not in checkpoint["split_manifest"]["split_ids"]["val"]:
        raise ValueError("Choose an explicitly saved validation ID; train/test fallback is disabled")
    candidates = [record for record in splits["val"] if record.sample_id == args.sample_id]
    if not candidates:
        raise ValueError("资料不足：该验证记录没有选定模态的有效输入")
    record = candidates[0]
    if "image" not in config["modalities"] or not record.image_path:
        raise ValueError("资料不足：该记录没有实际观测图像或模型未选择图像")
    assign_tct_ids(records, checkpoint["tct_vocab"])
    dataset = SysuDataset([record], args.images, train=False,
                          image_encoder=config["image_encoder"], modalities=config["modalities"])
    sample = dataset[0]
    batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value] for key, value in sample.items()}
    image_path = Path(args.images) / record.image_path
    with Image.open(image_path) as opened:
        original = opened.convert("RGB")
    geometry = image_geometry(original, config["image_encoder"])
    image_sha = file_sha256(image_path)
    expected_sha = next(entry["image_sha256"] for entry in checkpoint["split_manifest"]["records"]
                        if entry["sample_id"] == record.sample_id)
    if image_sha != expected_sha:
        raise ValueError("Image bytes changed after CPU manifest validation")
    device = torch.device(args.device)
    model, _ = restore_checkpoint(checkpoint, device)
    result = integrated_gradients(model, move_batch(batch, device), args.target, args.baseline, args.steps)
    if file_sha256(args.checkpoint) != checkpoint_sha or file_sha256(image_path) != image_sha:
        raise ValueError("Checkpoint or image bytes changed during explanation")
    output.mkdir(parents=True, exist_ok=False)
    pixels = result.pop("pixel_attribution")
    torch.save({"pixel_attribution": pixels, "shape": "[3,H,W] normalized model-input coordinates",
                "units": result["units"]}, output / "pixel_attribution.pt")
    display = save_heatmap(output / "image_ig.png", original, pixels, geometry, result["units"])
    write_json(output / "metadata.json", {**result, **display, "geometry": geometry,
               "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": checkpoint["epoch"],
               "split_manifest_sha256": checkpoint["split_manifest_sha256"], "split": "val",
               "sample_id": record.sample_id, "image_sha256": image_sha, "model_config": config,
               "pillow_version": pillow_version, "device": args.device,
               "fixed_nonimage_inputs_and_masks": True, "parameter_updates": False,
               "baseline_semantics": {"mean": "RGB ImageNet mean", "black": "RGB black normalized by ImageNet mean/std",
                                      "blur": "Gaussian-blurred same observed image; detail-relative response, not image removal"}[args.baseline],
               "limitation": "Numerical completeness is not faithfulness, lesion localization or medical causality"})
    print(f"Validation image IG written to {output}; review completeness and perturbation evidence separately")


if __name__ == "__main__":
    main()
