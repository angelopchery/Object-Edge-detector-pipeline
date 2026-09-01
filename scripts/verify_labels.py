"""Sanity-check YOLO label files against their images.

Checks:
  - every image has a label file, every label file has an image (orphans both ways)
  - all coordinates are normalised, within [0, 1]
  - no zero- or negative-area boxes
  - class IDs are only 0 or 1
Reports:
  - per-class instance counts
  - boxes per image (mean/min/max)
  - histogram of box areas as a fraction of frame area
  - boxes smaller than 0.1% of frame area flagged as suspicious

Exit code is non-zero if any hard check fails, so it can gate a commit.

Usage:
    # check one images/labels folder pair (e.g. before splitting)
    python scripts/verify_labels.py --images data/resized --labels data/labels_all

    # check every split of the assembled dataset
    python scripts/verify_labels.py --dataset data/dataset
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_CLASSES = {0, 1}
CLASS_NAMES = {0: "charger_brick", 1: "earphone_case"}
TINY_AREA_FRAC = 0.001  # 0.1% of frame area
AREA_BINS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 1.01]


def check_pair(images_dir: Path, labels_dir: Path, title: str) -> int:
    """Verify one images/labels folder pair. Returns the number of hard errors."""
    print(f"=== {title} ===")
    images = {p.stem: p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS}
    labels = {p.stem: p for p in sorted(labels_dir.glob("*.txt"))}

    errors = 0

    # Orphans in both directions.
    for stem in sorted(images.keys() - labels.keys()):
        print(f"  ERROR: image without label file: {images[stem].name}")
        errors += 1
    for stem in sorted(labels.keys() - images.keys()):
        print(f"  ERROR: label file without image: {labels[stem].name}")
        errors += 1

    class_counts: dict[int, int] = defaultdict(int)
    boxes_per_image: list[int] = []
    area_fracs: list[float] = []
    tiny_boxes: list[str] = []

    for stem in sorted(images.keys() & labels.keys()):
        lines = [ln.strip() for ln in labels[stem].read_text().splitlines() if ln.strip()]
        boxes_per_image.append(len(lines))
        for i, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                print(f"  ERROR: {stem}.txt line {i}: expected 5 fields, got {len(parts)}")
                errors += 1
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:])
            except ValueError:
                print(f"  ERROR: {stem}.txt line {i}: non-numeric field: {line!r}")
                errors += 1
                continue

            if cls not in VALID_CLASSES:
                print(f"  ERROR: {stem}.txt line {i}: invalid class ID {cls}")
                errors += 1
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                print(f"  ERROR: {stem}.txt line {i}: coordinate outside [0,1]: {line!r}")
                errors += 1
            if w <= 0.0 or h <= 0.0:
                print(f"  ERROR: {stem}.txt line {i}: zero/negative box area (w={w}, h={h})")
                errors += 1
                continue

            class_counts[cls] += 1
            area = w * h
            area_fracs.append(area)
            if area < TINY_AREA_FRAC:
                tiny_boxes.append(f"{stem}.txt line {i} (area={area * 100:.3f}% of frame)")

    print(f"  images: {len(images)}, label files: {len(labels)}")
    for cls in sorted(CLASS_NAMES):
        print(f"  instances of {CLASS_NAMES[cls]} ({cls}): {class_counts.get(cls, 0)}")
    if boxes_per_image:
        mean = sum(boxes_per_image) / len(boxes_per_image)
        print(f"  boxes per image: mean={mean:.2f}, min={min(boxes_per_image)}, max={max(boxes_per_image)}")

    if area_fracs:
        print("  box area as fraction of frame:")
        for lo, hi in zip(AREA_BINS, AREA_BINS[1:]):
            count = sum(1 for a in area_fracs if lo <= a < hi)
            bar = "#" * count
            print(f"    [{lo * 100:6.1f}%, {min(hi, 1.0) * 100:6.1f}%): {count:3d} {bar}")

    for msg in tiny_boxes:
        print(f"  SUSPICIOUS (<0.1% of frame): {msg}")

    print(f"  hard errors: {errors}\n")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify YOLO labels (orphans, ranges, areas, class IDs).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=Path, help="Dataset root with images/{split} and labels/{split}")
    group.add_argument("--images", type=Path, help="Single images folder (pair with --labels)")
    parser.add_argument("--labels", type=Path, help="Single labels folder (pair with --images)")
    args = parser.parse_args()

    total_errors = 0
    if args.dataset:
        for split in ("train", "val", "test"):
            img_dir = args.dataset / "images" / split
            lbl_dir = args.dataset / "labels" / split
            if img_dir.is_dir() and lbl_dir.is_dir():
                total_errors += check_pair(img_dir, lbl_dir, split)
            else:
                print(f"=== {split} === skipped (missing {img_dir} or {lbl_dir})\n")
    else:
        if not args.labels:
            parser.error("--images requires --labels")
        total_errors += check_pair(args.images, args.labels, f"{args.images} vs {args.labels}")

    if total_errors:
        print(f"FAILED: {total_errors} hard error(s). Fix them before training.")
        return 1
    print("OK: all hard checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
