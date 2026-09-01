"""Evaluate an ONNX model (FP32 or INT8) on a dataset split.

Computes precision, recall, mAP@0.5 and mAP@0.5:0.95 with a self-contained
matcher, so the FP32 and INT8 numbers are produced by the exact same code
path — not one from Ultralytics' validator and one from elsewhere.

Method:
  - inference through scripts/common.py preprocessing (same as parity check)
  - detections collected at a low confidence floor (0.001) so the PR curve
    is built over the full score range
  - greedy IoU matching per image/class, highest-confidence first, one
    detection per ground-truth box
  - AP = area under the interpolated PR curve (all-point interpolation),
    averaged over classes; mAP@0.5:0.95 averages IoU thresholds 0.5..0.95
  - reported precision/recall are taken at the best-F1 point of the
    IoU=0.5 PR curve (same convention Ultralytics uses)

Usage:
    python scripts/evaluate_onnx.py --onnx runs/detect/train/weights/best.onnx \
        --images data/dataset/images/val --labels data/dataset/labels/val
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from common import boxes_to_original, decode_and_nms, list_images, preprocess

# Measured from the makesense export (coordinate match vs CSV): 0=EarphoneCase, 1=ChargingCase.
CLASS_NAMES = {0: "earphone_case", 1: "charger_brick"}
IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)  # 0.50, 0.55, ..., 0.95


def load_gt(label_path: Path, orig_shape: tuple[int, int]) -> np.ndarray:
    """YOLO txt -> (N, 5) array [class, x1, y1, x2, y2] in original pixels."""
    if not label_path.is_file():
        return np.zeros((0, 5), dtype=np.float32)
    h, w = orig_shape
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = float(parts[0]), *(float(v) for v in parts[1:])
        rows.append([cls, (cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> tuple[float, np.ndarray, np.ndarray]:
    """All-point-interpolated AP. tp: per-detection hit flags at one IoU thr."""
    if n_gt == 0 or len(tp) == 0:
        return 0.0, np.zeros(0), np.zeros(0)
    order = conf.argsort()[::-1]
    tp = tp[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1 - tp)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)

    # Envelope + area under curve (all-point interpolation).
    r = np.concatenate(([0.0], recall, [1.0]))
    p = np.concatenate(([1.0], precision, [0.0]))
    p = np.maximum.accumulate(p[::-1])[::-1]
    idx = np.where(r[1:] != r[:-1])[0]
    ap = float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))
    return ap, precision, recall


def main() -> int:
    parser = argparse.ArgumentParser(description="P/R/mAP@0.5/mAP@0.5:0.95 for an ONNX model on a split.")
    parser.add_argument("--onnx", type=Path, required=True, help="ONNX model (FP32 or quantised)")
    parser.add_argument("--split", choices=("train", "val", "test"),
                        help="Shorthand: evaluate this split of --dataset")
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"),
                        help="Dataset root used with --split (default data/dataset)")
    parser.add_argument("--images", type=Path, help="Explicit images folder (instead of --split)")
    parser.add_argument("--labels", type=Path, help="Explicit labels folder (instead of --split)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size (default 640)")
    parser.add_argument("--iou-nms", type=float, default=0.45, help="NMS IoU threshold (default 0.45)")
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

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Per class: detection records (conf, iou-vs-matched-gt per threshold) and GT count.
    # We store per-detection TP flags for every IoU threshold at once.
    det_conf = {c: [] for c in CLASS_NAMES}
    det_tp = {c: [] for c in CLASS_NAMES}  # each entry: (len(IOU_THRESHOLDS),) 0/1
    n_gt = {c: 0 for c in CLASS_NAMES}

    for img_path in tqdm(images, desc=f"eval {args.onnx.name}"):
        tensor, gain, pad, orig_shape = preprocess(img_path, args.imgsz)
        raw = sess.run(None, {input_name: tensor})[0]
        dets = decode_and_nms(raw, conf_thres=0.001, iou_thres=args.iou_nms)
        if len(dets):
            dets[:, :4] = boxes_to_original(dets[:, :4], gain, pad, orig_shape)

        gt = load_gt(args.labels / (img_path.stem + ".txt"), orig_shape)
        for c in CLASS_NAMES:
            gt_c = gt[gt[:, 0] == c][:, 1:]
            dets_c = dets[dets[:, 5] == c]
            n_gt[c] += len(gt_c)
            if len(dets_c) == 0:
                continue
            # dets are already sorted by descending confidence (decode_and_nms).
            ious = iou_matrix(dets_c[:, :4], gt_c) if len(gt_c) else np.zeros((len(dets_c), 0))
            tp_flags = np.zeros((len(dets_c), len(IOU_THRESHOLDS)), dtype=np.float32)
            for t_idx, thr in enumerate(IOU_THRESHOLDS):
                matched_gt: set[int] = set()
                for d_idx in range(len(dets_c)):
                    if ious.shape[1] == 0:
                        continue
                    # Best still-unmatched GT box above the threshold, if any.
                    cand = [(ious[d_idx, g], g) for g in range(ious.shape[1])
                            if g not in matched_gt and ious[d_idx, g] >= thr]
                    if cand:
                        matched_gt.add(max(cand)[1])
                        tp_flags[d_idx, t_idx] = 1.0
            det_conf[c].extend(dets_c[:, 4].tolist())
            det_tp[c].extend(tp_flags)

    # Aggregate metrics.
    print(f"\nModel: {args.onnx}")
    print(f"Split: {args.images} ({len(images)} images)\n")
    print(f"| Class | GT | P | R | mAP@0.5 | mAP@0.5:0.95 |")
    print(f"|-------|----|----|----|---------|--------------|")

    map50s, map5095s = [], []
    for c, name in CLASS_NAMES.items():
        conf = np.array(det_conf[c])
        tps = np.array(det_tp[c]) if det_tp[c] else np.zeros((0, len(IOU_THRESHOLDS)))

        aps = [average_precision(tps[:, t], conf, n_gt[c])[0] for t in range(len(IOU_THRESHOLDS))]
        ap50, precision, recall = average_precision(tps[:, 0], conf, n_gt[c])
        ap5095 = float(np.mean(aps)) if aps else 0.0
        map50s.append(ap50)
        map5095s.append(ap5095)

        # P/R at the best-F1 point of the IoU=0.5 curve.
        if len(precision):
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            best = int(f1.argmax())
            p_best, r_best = float(precision[best]), float(recall[best])
        else:
            p_best = r_best = 0.0

        print(f"| {name} | {n_gt[c]} | {p_best:.3f} | {r_best:.3f} | {ap50:.3f} | {ap5095:.3f} |")

    print(f"| **all** | {sum(n_gt.values())} | - | - | "
          f"**{np.mean(map50s):.3f}** | **{np.mean(map5095s):.3f}** |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
