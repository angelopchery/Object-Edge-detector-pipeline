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
                     threshold: int, top_n: int) -> tuple[int, list[str]]:
    """Compare one eval split against train. Returns (flagged count, report lines)."""
    eval_images = list_images(eval_dir) if eval_dir.is_dir() else []
    if not eval_images:
        return 0, [f"=== {eval_name} vs train === skipped (no images in {eval_dir})", ""]

    closest = []  # (distance, eval_name, train_name)
    for img in eval_images:
        h = dhash(img)
        best = min(((hamming(h, th), tn) for tn, th in train_hashes.items()), key=lambda x: x[0])
        closest.append((best[0], img.name, best[1]))
    closest.sort()

    flagged = [c for c in closest if c[0] <= threshold]
    lines = [f"=== {eval_name} vs train === ({len(eval_images)} images)",
             f"  closest {min(top_n, len(closest))} pairs (Hamming distance out of 64, lower = more similar):"]
    for dist, ev, tr in closest[:top_n]:
        marker = "  <-- LEAK?" if dist <= threshold else ""
        lines.append(f"    {dist:2d}  {ev}  ~  {tr}{marker}")
    if flagged:
        lines.append(f"  {len(flagged)} pair(s) at or below threshold {threshold} — inspect these side by side.")
    else:
        lines.append(f"  no pair at or below threshold {threshold}.")
    lines.append("")
    return len(flagged), lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Perceptual-hash near-duplicate check across splits.")
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"),
                        help="Dataset root with images/{train,val,test} (default data/dataset)")
    parser.add_argument("--threshold", type=int, default=8,
                        help="Hamming distance at or below which a pair is flagged (default 8)")
    parser.add_argument("--top", type=int, default=10, help="Closest pairs to list per split (default 10)")
    parser.add_argument("--report", type=Path, default=None,
                        help="Also write the findings as markdown (e.g. notes/leakage_report.md)")
    args = parser.parse_args()

    train_dir = args.dataset / "images" / "train"
    train_images = list_images(train_dir) if train_dir.is_dir() else []
    if not train_images:
        print(f"ERROR: no train images in {train_dir}", file=sys.stderr)
        return 1
    train_hashes = {p.name: dhash(p) for p in train_images}

    flagged = 0
    all_lines: list[str] = []
    for split in ("val", "test"):
        n, lines = check_split_pair(split, args.dataset / "images" / split,
                                    train_hashes, args.threshold, args.top)
        flagged += n
        all_lines += lines

    verdict = (f"FLAGGED: {flagged} suspicious pair(s). Look at them before trusting any metric."
               if flagged else "OK: no near-duplicates found across splits at this threshold.")
    print("\n".join(all_lines))
    print(verdict)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("# Cross-split leakage audit\n\n"
                               f"Produced by `scripts/check_leakage.py --threshold {args.threshold}` "
                               f"(64-bit dHash, {len(train_hashes)} train images).\n\n```\n"
                               + "\n".join(all_lines) + verdict + "\n```\n")
        print(f"report written to {args.report}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
