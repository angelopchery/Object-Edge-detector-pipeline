"""Perceptual-hash leak check between dataset splits.

The scene-level split should prevent near-duplicates from crossing splits —
this script verifies that it actually did. It compares every val (and test)
image against every train image using a 64-bit difference hash and reports
the closest matches. A Hamming distance near 0 means visually near-identical
frames sit on both sides of a split: a leak, regardless of what the filenames
claim.

PLAN stage 7 gate: if val mAP@0.5 comes out suspiciously high (> 0.95), run
this before reporting anything.

Usage:
    python scripts/check_leakage.py --dataset data/dataset
    python scripts/check_leakage.py --dataset data/dataset --threshold 10
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from common import list_images


def dhash(image_path: Path) -> np.ndarray:
    """64-bit difference hash: resize to 9x8 grayscale, compare adjacent pixels."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read {image_path}")
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()  # 64 booleans


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def check_split_pair(eval_name: str, eval_dir: Path, train_hashes: dict[str, np.ndarray],
                     threshold: int, top_n: int) -> int:
    """Compare one eval split against train. Returns number of flagged pairs."""
    eval_images = list_images(eval_dir)
    if not eval_images:
        print(f"=== {eval_name} vs train === skipped (no images in {eval_dir})\n")
        return 0

    closest = []  # (distance, eval_name, train_name)
    for img in eval_images:
        h = dhash(img)
        best = min(((hamming(h, th), tn) for tn, th in train_hashes.items()), key=lambda x: x[0])
        closest.append((best[0], img.name, best[1]))
    closest.sort()

    flagged = [c for c in closest if c[0] <= threshold]
    print(f"=== {eval_name} vs train === ({len(eval_images)} images)")
    print(f"  closest {min(top_n, len(closest))} pairs (Hamming distance out of 64, lower = more similar):")
    for dist, ev, tr in closest[:top_n]:
        marker = "  <-- LEAK?" if dist <= threshold else ""
        print(f"    {dist:2d}  {ev}  ~  {tr}{marker}")
    if flagged:
        print(f"  {len(flagged)} pair(s) at or below threshold {threshold} — inspect these side by side.")
    else:
        print(f"  no pair at or below threshold {threshold}.")
    print()
    return len(flagged)


def main() -> int:
    parser = argparse.ArgumentParser(description="Perceptual-hash near-duplicate check across splits.")
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"),
                        help="Dataset root with images/{train,val,test} (default data/dataset)")
    parser.add_argument("--threshold", type=int, default=8,
                        help="Hamming distance at or below which a pair is flagged (default 8)")
    parser.add_argument("--top", type=int, default=10, help="Closest pairs to list per split (default 10)")
    args = parser.parse_args()

    train_dir = args.dataset / "images" / "train"
    train_images = list_images(train_dir) if train_dir.is_dir() else []
    if not train_images:
        print(f"ERROR: no train images in {train_dir}", file=sys.stderr)
        return 1
    train_hashes = {p.name: dhash(p) for p in train_images}

    flagged = 0
    for split in ("val", "test"):
        flagged += check_split_pair(split, args.dataset / "images" / split,
                                    train_hashes, args.threshold, args.top)

    if flagged:
        print(f"FLAGGED: {flagged} suspicious pair(s). Look at them before trusting any metric.")
        return 1
    print("OK: no near-duplicates found across splits at this threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
