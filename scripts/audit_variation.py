"""Capture-variation audit (PLAN Phase 2c).

Measures how much lighting, background, and distance variation the dataset
really contains, instead of guessing from the 14-minute capture window.
Writes a markdown report for the README/Known Gaps to quote.

Metrics:
  - brightness: mean grayscale value per image; spread across the dataset
  - background diversity: coarse HSV histogram of the pixels OUTSIDE all
    boxes, greedily grouped by histogram correlation — a proxy for "how many
    distinct backgrounds exist"
  - box areas as fraction of frame — distance variation proxy
  - orientation and aspect-ratio mix
  - per-class instance counts, boxes per image

Usage:
    python scripts/audit_variation.py --images data/prepared --labels YoloLabels --out notes/variation_report.md
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from common import list_images

CLASS_NAMES = {0: "earphone_case", 1: "charger_brick"}
BG_CORR_THRESHOLD = 0.55  # histogram correlation above this = "same background group"


def load_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.is_file():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            out.append((int(p[0]), *(float(v) for v in p[1:])))
    return out


def background_hist(img: np.ndarray, boxes) -> np.ndarray:
    """Coarse HSV histogram of pixels outside all labelled boxes."""
    h, w = img.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    for _, cx, cy, bw, bh in boxes:
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        mask[max(y1, 0):y2, max(x1, 0):x2] = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, None).flatten()


def group_backgrounds(hists: list[np.ndarray]) -> list[int]:
    """Greedy grouping: join the first existing group whose representative
    correlates above threshold; otherwise start a new group."""
    reps: list[np.ndarray] = []
    labels = []
    for h in hists:
        for gi, rep in enumerate(reps):
            if cv2.compareHist(rep.astype(np.float32), h.astype(np.float32),
                               cv2.HISTCMP_CORREL) > BG_CORR_THRESHOLD:
                labels.append(gi)
                break
        else:
            reps.append(h)
            labels.append(len(reps) - 1)
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure lighting/background/distance variation.")
    parser.add_argument("--images", type=Path, required=True, help="Images folder (resized ok)")
    parser.add_argument("--labels", type=Path, required=True, help="YOLO labels folder")
    parser.add_argument("--out", type=Path, required=True, help="Markdown report output path")
    args = parser.parse_args()

    images = list_images(args.images)
    if not images:
        print(f"ERROR: no images in {args.images}", file=sys.stderr)
        return 1

    brightness, bg_hists, areas, orientations, n_boxes = [], [], [], [], []
    cls_counts = {0: 0, 1: 0}
    for p in images:
        img = cv2.imread(str(p))
        boxes = load_boxes(args.labels / (p.stem + ".txt"))
        brightness.append(float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()))
        bg_hists.append(background_hist(img, boxes))
        h, w = img.shape[:2]
        orientations.append("portrait" if h >= w else "landscape")
        n_boxes.append(len(boxes))
        for cls, _, _, bw, bh in boxes:
            areas.append(bw * bh)
            cls_counts[cls] = cls_counts.get(cls, 0) + 1

    b = np.array(brightness)
    a = np.array(areas)
    groups = group_backgrounds(bg_hists)
    n_groups = len(set(groups))
    group_sizes = sorted(np.bincount(groups), reverse=True)
    area_span = float(a.max() / max(a.min(), 1e-9))

    hist_edges = np.histogram(b, bins=8, range=(0, 255))[0]
    lines = [
        "# Capture variation report",
        "",
        f"Produced by `scripts/audit_variation.py` over {len(images)} images.",
        "",
        "## Brightness (grayscale mean per image, 0-255)",
        f"- mean {b.mean():.1f}, std {b.std():.1f}, min {b.min():.1f}, max {b.max():.1f}",
        "- histogram (8 bins over 0-255): " + " ".join(str(c) for c in hist_edges),
        "",
        "## Background diversity (HSV hist of non-box pixels, correlation-grouped)",
        f"- {n_groups} background group(s) at correlation threshold {BG_CORR_THRESHOLD}",
        f"- group sizes: {group_sizes}",
        "",
        "## Box areas (fraction of frame) — distance-variation proxy",
        f"- min {a.min() * 100:.2f}%, median {np.median(a) * 100:.2f}%, max {a.max() * 100:.2f}%",
        f"- max/min span: {area_span:.1f}x",
        "",
        "## Composition",
        f"- orientation mix: {orientations.count('portrait')} portrait / {orientations.count('landscape')} landscape",
        f"- boxes per image: mean {np.mean(n_boxes):.2f}, min {min(n_boxes)}, max {max(n_boxes)}",
        f"- instances: {CLASS_NAMES[0]}={cls_counts.get(0, 0)}, {CLASS_NAMES[1]}={cls_counts.get(1, 0)}",
        "",
        "## Decision inputs (PLAN Phase 2c table)",
        f"- brightness std: {b.std():.1f} (wide if >~25)",
        f"- background groups: {n_groups} (threshold in table: >=3)",
        f"- box-area span: {area_span:.1f}x (threshold in table: >=10x = order of magnitude)",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nReport written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
