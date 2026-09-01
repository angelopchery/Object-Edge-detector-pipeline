"""Pre-label unannotated images with a trained model.

Writes YOLO-format .txt files (save_txt=True) for import into makesense.ai.
Every pre-labelled box gets manually reviewed and corrected before it enters
the dataset — this script only produces a starting point, not ground truth.

Usage:
    python scripts/prelabel.py --weights runs/detect/seed_run/weights/best.pt --source data/resized_unlabelled
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-label images with a trained model (YOLO txt output).")
    parser.add_argument("--weights", type=Path, required=True, help="Trained model, e.g. best.pt")
    parser.add_argument("--source", type=Path, required=True, help="Folder of images to pre-label")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold (default 0.30)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size (default 640)")
    parser.add_argument("--name", default="prelabel", help="Run name under runs/detect/")
    parser.add_argument("--save-conf", action="store_true",
                        help="Append confidence to each line (default off: makesense.ai expects 5 fields)")
    args = parser.parse_args()

    if not args.weights.is_file():
        print(f"ERROR: weights not found: {args.weights}", file=sys.stderr)
        return 1
    if not args.source.is_dir():
        print(f"ERROR: source folder not found: {args.source}", file=sys.stderr)
        return 1

    model = YOLO(str(args.weights))
    results = model.predict(
        source=str(args.source),
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,        # annotated previews, handy while reviewing
        save_txt=True,
        save_conf=args.save_conf,
        name=args.name,
        project="runs/detect",
    )

    n_boxes = sum(len(r.boxes) for r in results)
    save_dir = Path(results[0].save_dir) if results else None
    print(f"\nPre-labelled {len(results)} images, {n_boxes} boxes at conf>={args.conf}")
    if save_dir:
        print(f"Label txts:  {save_dir / 'labels'}")
        print(f"Previews:    {save_dir}")
    print("Import the txts into makesense.ai and review EVERY box before accepting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
