"""Render YOLO label boxes onto their images for visual review.

The cheapest, most effective annotation check there is: draw every box and
look at it. Run after each annotation batch and after pre-labelling — a
mislabelled or mirrored box is unmissable once rendered.

Usage:
    python scripts/render_labels.py --images data/prepared --labels data/labels_all --out runs/render_labels
"""

import argparse
import sys
from pathlib import Path

import cv2

from common import list_images

CLASS_NAMES = {0: "charger_brick", 1: "earphone_case"}
CLASS_COLORS = {0: (0, 200, 0), 1: (255, 128, 0)}  # BGR: green brick, blue-ish case
UNKNOWN_COLOR = (0, 0, 255)  # red for invalid class IDs — should never appear


def draw_labels(img, label_path: Path):
    h, w = img.shape[:2]
    n = 0
    if not label_path.is_file():
        cv2.putText(img, "NO LABEL FILE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, UNKNOWN_COLOR, 2)
        return img, n
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        color = CLASS_COLORS.get(cls, UNKNOWN_COLOR)
        name = CLASS_NAMES.get(cls, f"class {cls}?!")
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        n += 1
    return img, n


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw YOLO label boxes on images for eyeball review.")
    parser.add_argument("--images", type=Path, required=True, help="Images folder")
    parser.add_argument("--labels", type=Path, required=True, help="YOLO label folder")
    parser.add_argument("--out", type=Path, required=True, help="Output folder for rendered images")
    args = parser.parse_args()

    images = list_images(args.images)
    if not images:
        print(f"ERROR: no images in {args.images}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARNING: could not read {img_path.name}, skipped")
            continue
        img, n = draw_labels(img, args.labels / (img_path.stem + ".txt"))
        total_boxes += n
        cv2.imwrite(str(args.out / img_path.name), img)

    print(f"Rendered {len(images)} images ({total_boxes} boxes) into {args.out}")
    print("Scroll through them. Every box must sit tight on its object.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
