"""Build a protocol from externally reviewed, anonymous patient identity mapping."""

import argparse
import json
from pathlib import Path

from opmd_pred.data.sysu_protocol import build_manifest, manifest_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--groups", required=True, help="Reviewed anonymous CSV: sample_id,patient_id; do not substitute sample_id for patients")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a frozen split manifest")
    manifest = build_manifest(args.csv, args.images, args.groups, args.seed, args.val_fraction, args.test_fraction)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"splits": manifest_summary(manifest), "excluded_count": len(manifest["excluded"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
