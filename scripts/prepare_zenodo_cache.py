import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import PILToTensor, Resize
from tqdm import tqdm

from opmd_pred.data.zenodo import _attach_patient_ids, _load_rows, _patient_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/zenodo/Imagewise_Data.csv")
    parser.add_argument("--images", default="data/zenodo/Images")
    parser.add_argument("--output", default="outputs/zenodo_cache.pt")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = _load_rows(args.csv)
    _attach_patient_ids(rows, _patient_ids(args.csv))
    image_paths = {
        path.stem: path
        for path in Path(args.images).rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    resize = Resize((224, 224))
    to_tensor = PILToTensor()
    image_names = [row["Image Name"] for row in rows]

    def prepare_image(image_name):
        with Image.open(image_paths[image_name]) as source:
            image = source.convert("RGB")
        return to_tensor(resize(image))

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        images = list(
            tqdm(
                executor.map(prepare_image, image_names),
                total=len(image_names),
                desc="Preparing Zenodo images",
                unit="image",
            )
        )
    indices = {image_name: index for index, image_name in enumerate(image_names)}
    cache = {"images": torch.stack(images), "indices": indices}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    print(f"Saved {len(images)} images to {output}")


if __name__ == "__main__":
    main()
