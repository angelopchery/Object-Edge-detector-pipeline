"""Verify that the exported ONNX model matches the PyTorch model numerically.

Method (this is *how* parity is confirmed, not just that it was):
  1. The same preprocessed tensor is fed to both the PyTorch model (eval mode,
     FP32, no augmentation) and ONNX Runtime.
  2. Raw output tensors are compared element-wise: max and mean absolute diff.
  3. Both raw outputs then go through the SAME decode+NMS (scripts/common.py),
     and final box coordinates are compared in letterbox pixels.
  4. Repeated over N validation images; the worst case across all images is
     reported. Exit code is non-zero if worst raw max-abs-diff > --tolerance.

Usage:
    python scripts/verify_parity.py --weights runs/detect/train/weights/best.pt \
        --onnx runs/detect/train/weights/best.onnx --images data/dataset/images/val
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

from common import decode_and_nms, list_images, preprocess


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PyTorch vs ONNX outputs (raw + post-NMS).")
    parser.add_argument("--weights", type=Path, required=True, help="PyTorch .pt model")
    parser.add_argument("--onnx", type=Path, required=True, help="Exported .onnx model")
    parser.add_argument("--images", type=Path, required=True, help="Folder of validation images")
    parser.add_argument("--num-images", type=int, default=10, help="Images to test (default 10)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size (default 640)")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Max raw abs diff allowed (default 1e-3)")
    args = parser.parse_args()

    images = list_images(args.images)[: args.num_images]
    if not images:
        print(f"ERROR: no images found in {args.images}", file=sys.stderr)
        return 1

    # PyTorch model in eval mode, FP32 on CPU: ONNX export is FP32, so parity
    # must be checked in FP32 too (CUDA AMP kernels would add unrelated noise).
    torch_model = YOLO(str(args.weights)).model.float().eval()
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    worst_raw_max = worst_raw_mean = worst_box = 0.0
    worst_raw_image = worst_box_image = "-"

    print(f"{'image':30s} {'raw max diff':>14s} {'raw mean diff':>14s} {'box coord diff px':>18s}")
    for img_path in images:
        tensor, _, _, _ = preprocess(img_path, args.imgsz)

        with torch.no_grad():
            out = torch_model(torch.from_numpy(tensor))
        # In eval, the detect head returns (predictions, aux); take predictions.
        pt_raw = (out[0] if isinstance(out, (list, tuple)) else out).numpy()
        ort_raw = sess.run(None, {input_name: tensor})[0]

        raw_max = float(np.max(np.abs(pt_raw - ort_raw)))
        raw_mean = float(np.mean(np.abs(pt_raw - ort_raw)))

        pt_dets = decode_and_nms(pt_raw)
        ort_dets = decode_and_nms(ort_raw)
        if len(pt_dets) != len(ort_dets):
            box_diff = float("inf")  # detection-count mismatch is a hard parity failure
            box_str = f"COUNT {len(pt_dets)} vs {len(ort_dets)}"
        elif len(pt_dets) == 0:
            box_diff, box_str = 0.0, "no detections"
        else:
            box_diff = float(np.max(np.abs(pt_dets[:, :4] - ort_dets[:, :4])))
            box_str = f"{box_diff:.6f}"

        print(f"{img_path.name:30s} {raw_max:>14.2e} {raw_mean:>14.2e} {box_str:>18s}")

        if raw_max > worst_raw_max:
            worst_raw_max, worst_raw_image = raw_max, img_path.name
        worst_raw_mean = max(worst_raw_mean, raw_mean)
        if box_diff > worst_box:
            worst_box, worst_box_image = box_diff, img_path.name

    print(f"\nWorst case over {len(images)} images:")
    print(f"  raw max abs diff:  {worst_raw_max:.2e}  ({worst_raw_image})")
    print(f"  raw mean abs diff: {worst_raw_mean:.2e}")
    print(f"  post-NMS box coordinate diff: {worst_box:.6f} px  ({worst_box_image})")

    if worst_raw_max > args.tolerance:
        print(f"FAILED: raw diff exceeds tolerance {args.tolerance:.0e}")
        return 1
    print(f"OK: within tolerance {args.tolerance:.0e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
