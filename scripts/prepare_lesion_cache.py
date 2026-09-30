"""根据 Annotation.json 的病灶区域生成"聚焦缓存"。

-  每张图取 category_id=1(Lesion) 里面积最大的 bbox 作为焦点；若该图无病灶框，
   退回 category_id=2(Oral Cavity)；再无则整图中心。
-  bbox 略做 padding，裁到病灶，再 Resize 到指定 size。
-  输出 dict 结构与 prepare_zenodo_cache.py 一致：{"images":Tensor[N,3,S,S], "indices":{name:idx}}。
-  训练/推理都用同一份聚焦缓存，检验"病灶尺度纹理分类"能否超整图基线。
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import PILToTensor, Resize
from tqdm import tqdm

from opmd_pred.data.zenodo import _load_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/zenodo/Imagewise_Data.csv")
    parser.add_argument("--images", default="data/zenodo/Images")
    parser.add_argument("--annotation", default="data/zenodo/Annotation.json")
    parser.add_argument("--output", default="outputs/zenodo_lesion_cache.pt")
    parser.add_argument("--size", type=int, default=288)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pad-frac", type=float, default=0.15)
    return parser.parse_args()


def focus_box(annotations, image_id, pad_frac):
    """返回该图的病灶框(左上x,y,框w,h 已含padding)。无病灶退回口腔区。"""
    lesion = [a for a in annotations if a["image_id"] == image_id and a["category_id"] == 1]
    source = [a for a in annotations if a["image_id"] == image_id and a["category_id"] == 2]
    ann = max(lesion or source or [], key=lambda a: a["area"])
    if not ann:
        return None
    x, y, w, h = ann["bbox"]
    pad_x, pad_y = w * pad_frac, h * pad_frac
    return (x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y)


def main():
    args = parse_args()
    rows = _load_rows(args.csv)
    annotation = json.load(open(args.annotation, encoding="utf-8"))
    image_paths = {
        path.stem: path
        for path in Path(args.images).rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    name_to_id = {im["file_name"]: im["id"] for im in annotation["images"]}
    resize = Resize((args.size, args.size))
    to_tensor = PILToTensor()
    image_names = [row["Image Name"] for row in rows]

    def prepare(image_name):
        with Image.open(image_paths[image_name]) as source:
            image = source.convert("RGB")
        w, h = image.size
        box = focus_box(annotation["annotations"], name_to_id[image_name + ".jpg"], args.pad_frac)
        if box is None:
            image = image.resize((args.size, args.size))
        else:
            x, y, bw, bh = box
            left = max(int(x), 0)
            top = max(int(y), 0)
            right = min(int(x + bw), w)
            bottom = min(int(y + bh), h)
            if right - left < 2 or bottom - top < 2:
                image = image.resize((args.size, args.size))
            else:
                image = image.crop((left, top, right, bottom))
                image = resize(image)
        return to_tensor(image)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        images = list(
            tqdm(executor.map(prepare, image_names), total=len(image_names), desc="lesion-crop", unit="image")
        )
    indices = {name: index for index, name in enumerate(image_names)}
    cache = {"images": torch.stack(images), "indices": indices}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    print(f"Saved {len(images)} focused images to {output}")


if __name__ == "__main__":
    main()