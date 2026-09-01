"""Render ONNX model predictions vs ground truth for failure analysis.

Ground truth is drawn in green, predictions in red with class and confidence.
Output images go into --out, sorted so the worst images (most GT boxes with
no matching prediction, plus false positives) are easy to find: each filename
is prefixed with its error count, so an ascending sort surfaces the failures.

PLAN stage 13: pick the worst images from this output for the A4 write-up.

Usage:
    python scripts/render_predictions.py --onnx models/best.onnx --split val --out runs/render_val
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from common import boxes_to_original, decode_and_nms, list_images, preprocess

CLASS_NAMES = {0: "charger_brick", 1: "earphone_case"}
GT_COLOR = (0, 200, 0)      # green
PRED_COLOR = (0, 0, 255)    # red
MATCH_IOU = 0.5


def load_gt(label_path: Path, shape) -> np.ndarray:
    """YOLO txt -> (N, 5) [class, x1, y1, x2, y2] in pixels."""
    if not label_path.is_file():
        return np.zeros((0, 5), dtype=np.float32)
    h, w = shape
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            cls, cx, cy, bw, bh = float(parts[0]), *(float(v) for v in parts[1:])
            rows.append([cls, (cx - bw / 2) * w, (cy - bh / 2) * h,
                         (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def iou_one(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / (union + 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render predictions (red) vs ground truth (green).")
    parser.add_argument("--onnx", type=Path, required=True, help="ONNX model")
    parser.add_argument("--split", choices=("train", "val", "test"), help="Dataset split shorthand")
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"), help="Dataset root")
    parser.add_argument("--images", type=Path, help="Explicit images folder (instead of --split)")
    parser.add_argument("--labels", type=Path, help="Explicit labels folder (instead of --split)")
    parser.add_argument("--out", type=Path, required=True, help="Output folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size (default 640)")
    args = parser.parse_args()

    if args.split:
        args.images = args.dataset / "images" / args.split
        args.labels = args.dataset / "labels" / args.split
    elif not (args.images and args.labels):
        parser.error("provide either --split or both --images and --labels")

    images = list_images(args.images)
    if not images:
        print(f"ERROR: no images in {args.images}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    summary = []
    for img_path in images:
        tensor, gain, pad, orig_shape = preprocess(img_path, args.imgsz)
        raw = sess.run(None, {input_name: tensor})[0]
        dets = decode_and_nms(raw, conf_thres=args.conf)
        if len(dets):
            dets[:, :4] = boxes_to_original(dets[:, :4], gain, pad, orig_shape)
        gt = load_gt(args.labels / (img_path.stem + ".txt"), orig_shape)

        # Count errors: GT boxes with no same-class prediction at IoU>=0.5 (misses)
        # plus predictions matching no GT box (false positives).
        misses = sum(1 for g in gt if not any(
            d[5] == g[0] and iou_one(g[1:], d[:4]) >= MATCH_IOU for d in dets))
        false_pos = sum(1 for d in dets if not any(
            d[5] == g[0] and iou_one(g[1:], d[:4]) >= MATCH_IOU for g in gt))
        errors = misses + false_pos

        img = cv2.imread(str(img_path))
        for g in gt:
            x1, y1, x2, y2 = (int(v) for v in g[1:])
            cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 2)
            cv2.putText(img, f"GT {CLASS_NAMES.get(int(g[0]), '?')}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1)
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, 2)
            cv2.putText(img, f"{CLASS_NAMES.get(int(d[5]), '?')} {d[4]:.2f}",
                        (x1, min(y2 + 16, orig_shape[0] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_COLOR, 1)

        out_name = f"err{errors:02d}_{img_path.name}"
        cv2.imwrite(str(args.out / out_name), img)
        summary.append((errors, misses, false_pos, img_path.name))

    summary.sort(reverse=True)
    print(f"Rendered {len(images)} images into {args.out} (green=GT, red=prediction)")
    print(f"Filenames are prefixed errNN_ — sort descending to find the worst.\n")
    print("Worst 5 images:")
    for errors, misses, fps, name in summary[:5]:
        print(f"  {name}: {misses} missed GT box(es), {fps} false positive(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
