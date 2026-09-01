"""EXIF orientation gate (PLAN Phase 2a).

makesense.ai labelled the images as the browser displayed them — with EXIF
rotation applied. The normalised YOLO labels are only valid against pixels in
that same orientation. This script compares, for every image:

    ImageOps.exif_transpose(Image.open(p)).size   (labelling orientation)
    vs the image_width/image_height recorded in YoloCSV.csv

and also reports the raw (un-transposed) size, so it is explicit whether the
files carry an EXIF rotation that downstream code must apply.

Usage:
    python scripts/check_exif_orientation.py --images YoloData --csv YoloCSV.csv
"""

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageOps


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify EXIF-applied dimensions match the CSV labelling dims.")
    parser.add_argument("--images", type=Path, required=True, help="Folder of original images")
    parser.add_argument("--csv", type=Path, required=True, help="makesense CSV export with image dims")
    args = parser.parse_args()

    csv_dims: dict[str, tuple[int, int]] = {}
    with args.csv.open() as f:
        for row in csv.DictReader(f):
            csv_dims[row["image_name"]] = (int(row["image_width"]), int(row["image_height"]))

    checked = mismatches = exif_rotated = 0
    for name, (cw, ch) in sorted(csv_dims.items()):
        path = args.images / name
        if not path.is_file():
            print(f"  ERROR: image in CSV but not on disk: {name}")
            mismatches += 1
            continue
        with Image.open(path) as im:
            raw = im.size
            transposed = ImageOps.exif_transpose(im).size
        checked += 1
        if raw != transposed:
            exif_rotated += 1
        if transposed != (cw, ch):
            print(f"  MISMATCH: {name}: exif-applied {transposed} != CSV {(cw, ch)} (raw {raw})")
            mismatches += 1

    print(f"\nchecked {checked} images against CSV dims")
    print(f"images whose EXIF rotates them: {exif_rotated} "
          f"(resize MUST apply exif_transpose{' — it does' if exif_rotated else ''})")
    if mismatches:
        print(f"FAILED: {mismatches} mismatch(es) — labels and pixels are in different orientations.")
        return 1
    print("OK: every image's EXIF-applied size matches the labelling dimensions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
