"""Run the ONNX model over a folder of images and render the detections.

The deployment artifact path: models/best.onnx through the exact
scripts/common.py pipeline used by the benchmark and evaluation — not the
Ultralytics predictor — so what you see is what ships.

Usage:
    python scripts/detect_folder.py --onnx models/best.onnx --source <folder> --out runs/detect_demo
"""

import argparse
import sys
from pathlib import Path

import cv2
import onnxruntime as ort

from common import boxes_to_original, decode_and_nms, list_images, preprocess

CLASS_NAMES = {0: "earphone_case", 1: "charger_brick"}
COLORS = {0: (0, 200, 0), 1: (255, 128, 0)}  # BGR


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX inference + rendered boxes for a folder.")
    parser.add_argument("--onnx", type=Path, default=Path("models/best.onnx"), help="ONNX model")
    parser.add_argument("--source", type=Path, required=True, help="Folder of images")
    parser.add_argument("--out", type=Path, required=True, help="Output folder for rendered images")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size (default 640)")
    args = parser.parse_args()

    images = list_images(args.source)
    if not images:
        print(f"ERROR: no images in {args.source}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    for img_path in images:
        tensor, gain, pad, orig_shape = preprocess(img_path, args.imgsz)
        raw = sess.run(None, {input_name: tensor})[0]
        dets = decode_and_nms(raw, conf_thres=args.conf)
        if len(dets):
            dets[:, :4] = boxes_to_original(dets[:, :4], gain, pad, orig_shape)

        img = cv2.imread(str(img_path))
        thickness = max(2, orig_shape[0] // 400)
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d[:4])
            cls, conf = int(d[5]), float(d[4])
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS.get(cls, (0, 0, 255)), thickness)
            cv2.putText(img, f"{CLASS_NAMES.get(cls, '?')} {conf:.2f}", (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, thickness / 3, COLORS.get(cls, (0, 0, 255)), thickness)

        found = ", ".join(f"{CLASS_NAMES[int(d[5])]} {d[4]:.2f}" for d in dets) or "nothing"
        print(f"{img_path.name}: {found}")
        cv2.imwrite(str(args.out / (img_path.stem + "_pred.jpg")), img)

    print(f"\nRendered results in {args.out} — open the folder and scroll.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
