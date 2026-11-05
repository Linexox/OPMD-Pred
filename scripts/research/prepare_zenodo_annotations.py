"""Build hash-pinned photo/classification/multi-box detection records."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

from opmd_pred.data.sysu_protocol import file_sha256
from opmd_pred.data.zenodo import CATEGORIES


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "labels", "annotations", "split-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--corrections", type=Path)
    args = parser.parse_args()
    source = json.loads(args.annotations.read_text(encoding="utf-8"))
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    corrections = json.loads(args.corrections.read_text(encoding="utf-8")) if args.corrections else {}
    decisions = corrections.get("annotations", {})
    source_images = {Path(i["file_name"]).stem: i for i in source["images"]}
    annotations = defaultdict(list)
    for a in source["annotations"]:
        if a["category_id"] == 1:
            annotations[a["image_id"]].append(a)
    paths = {}
    for path in args.images.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if path.stem in paths:
                raise ValueError("Ambiguous source stem " + path.stem)
            paths[path.stem] = path
    with args.labels.open(encoding="utf-8-sig", newline="") as stream:
        labels = {r["Image Name"]: r["Category"].strip() for r in csv.DictReader(stream)}
    records, reviews, groups, hashes = [], [], {}, {}
    for split, expected in (("train", 2366), ("validation", 315), ("test", 313)):
        rows = manifest["splits"][split]
        assert len(rows) == expected
        groups[split], hashes[split] = set(), set()
        for row in rows:
            name = row["image_name"]
            path = paths[name]
            digest = file_sha256(path)
            assert digest == row["sha256"] and labels[name] == row["category"]
            groups[split].add(row["patient_id"])
            hashes[split].add(digest)
            with Image.open(path) as image:
                width, height = image.size
                orientation = image.getexif().get(274, 1)
            image_id = source_images[name]["id"]
            boxes, bad = [], []
            original = annotations[image_id]
            for a in original:
                x, y, w, h = a["bbox"]
                candidate = [x, y, x+w, y+h]
                if corrections.get('coordinate_frame') == 'exif_display' and orientation in (3,6,8):
                    # Annotation UI used the EXIF-oriented display. Training retains native
                    # pixels to match the first classifier; map all four corners back.
                    corners=[(x,y),(x+w,y),(x,y+h),(x+w,y+h)]
                    if orientation==6:
                        corners=[(cy,height-cx) for cx,cy in corners]
                    elif orientation==8:
                        corners=[(width-cy,cx) for cx,cy in corners]
                    else:
                        corners=[(width-cx,height-cy) for cx,cy in corners]
                    candidate=[min(c[0] for c in corners),min(c[1] for c in corners),
                               max(c[0] for c in corners),max(c[1] for c in corners)]
                action = decisions.get(str(a["id"]))
                if action:
                    assert action.get("reason") and action["original_bbox"] == a["bbox"]
                    if action["action"] == "replace":
                        candidate = action["xyxy"]
                    elif action["action"] == "mask_image":
                        bad.append(a["id"])
                    else:
                        raise ValueError("Unsupported annotation decision")
                valid = (len(candidate) == 4 and 0 <= candidate[0] < candidate[2] <= width
                         and 0 <= candidate[1] < candidate[3] <= height)
                if not valid or (action and action["action"] == "mask_image"):
                    bad.append(a["id"])
                    reviews.append({"image_name": name, "split": split, "annotation_id": a["id"],
                        "width": width, "height": height, "original_bbox": json.dumps(a["bbox"]),
                        "exif_orientation": orientation, "decision": action["action"] if action else "unresolved",
                        "reason": action["reason"] if action else "invalid_or_out_of_bounds"})
                if valid:
                    boxes.append([float(v) for v in candidate])
            healthy = row["category"] == "Healthy"
            detection_valid = not bad and ((healthy and not original) or (not healthy and bool(boxes)))
            records.append({"image_name": name, "image_id": image_id, "file_name": path.name,
                "relative_path": path.relative_to(args.images).as_posix(), "split": split,
                "patient_id": row["patient_id"], "category": row["category"], "label": CATEGORIES[row["category"]],
                "sha256": digest, "width": width, "height": height, "boxes": boxes,
                "annotation_ids": [a["id"] for a in original], "unresolved_annotation_ids": sorted(set(bad)),
                "classification_valid": True, "roi_valid": detection_valid and bool(boxes),
                "detection_valid": detection_valid, "exif_orientation": orientation,
                "coordinate_transform": 'exif_display_to_native_pixels' if
                    corrections.get('coordinate_frame')=='exif_display' and orientation in (3,6,8) else 'identity'})
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not groups[left] & groups[right] and not hashes[left] & hashes[right]
    assert len(records) == len({r["image_name"] for r in records}) == 2994
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "images.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
    with (args.output / "annotation_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["image_name", "split", "annotation_id", "width", "height", "original_bbox", "exif_orientation", "decision", "reason"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reviews)
    for split in manifest["splits"]:
        split_records = [r for r in records if r["split"] == split and r["detection_valid"]]
        coco = {"info": {"description": "Validated class-agnostic oral lesion boxes"},
                "categories": [{"id": 1, "name": "Lesion"}], "images": [], "annotations": []}
        for r in split_records:
            coco["images"].append({"id": r["image_id"], "file_name": r["relative_path"],
                                   "width": r["width"], "height": r["height"]})
            for b in r["boxes"]:
                x1, y1, x2, y2 = b
                coco["annotations"].append({"id": len(coco["annotations"])+1, "image_id": r["image_id"],
                    "category_id": 1, "bbox": [x1, y1, x2-x1, y2-y1], "area": (x2-x1)*(y2-y1), "iscrowd": 0})
        write(args.output / f"{split}.coco.json", coco)
    selected = [r for r in records if r["unresolved_annotation_ids"]]
    # Deterministic healthy samples are included for checking empty-target handling.
    selected += [r for r in records if r["category"] == "Healthy"][:8]
    preview_dir = args.output / "previews"
    preview_dir.mkdir(exist_ok=True)
    for r in selected:
        with Image.open(paths[r["image_name"]]) as image:
            image = image.convert("RGB")
            scale = min(900/image.width, 700/image.height, 1.)
            image = image.resize((round(image.width*scale), round(image.height*scale)))
        draw = ImageDraw.Draw(image)
        for x1,y1,x2,y2 in r['boxes']:
            draw.rectangle([x1*scale,y1*scale,x2*scale,y2*scale],outline='red',width=3)
        image.save(preview_dir / (r["image_name"] + ".jpg"))
    write(args.output / "annotation_audit.json", {"records": len(records),
        "split_counts": dict(Counter(r["split"] for r in records)),
        "class_counts": {s: dict(Counter(r["category"] for r in records if r["split"] == s)) for s in manifest["splits"]},
        "masked_detection_images": len({r["image_name"] for r in reviews}),
        "annotation_review_count": len(reviews), "unresolved_decisions": sum(r["decision"] == "unresolved" for r in reviews),
        "healthy_no_boxes": sum(r["category"] == "Healthy" and not r["boxes"] for r in records),
        "source_sha256": {str(p): file_sha256(p) for p in (args.annotations, args.labels, args.split_manifest)},
        "images_jsonl_sha256": file_sha256(args.output / "images.jsonl"),
        "all_source_images_verified": True, "patient_and_source_hash_disjoint": True,
        "corrections": corrections})
    print((args.output / "annotation_audit.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
