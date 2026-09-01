"""Export a trained .pt model to ONNX (opset 12, fixed batch 1, simplified).

Prints the exported graph's input/output tensor names and shapes so the
README can quote them.

Usage:
    python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

import onnx
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Export best.pt to ONNX (opset 12, batch 1, simplify).")
    parser.add_argument("--weights", type=Path, required=True, help="Trained model .pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size (default 640)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset (default 12)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Directory to copy the deliverables into as best.pt/best.onnx (e.g. models)")
    args = parser.parse_args()

    if not args.weights.is_file():
        print(f"ERROR: weights not found: {args.weights}", file=sys.stderr)
        return 1

    model = YOLO(str(args.weights))
    onnx_path = model.export(
        format="onnx",
        opset=args.opset,
        imgsz=args.imgsz,
        dynamic=False,   # fixed batch 1
        simplify=True,
        batch=1,
    )

    graph = onnx.load(onnx_path).graph

    def shape_of(value_info) -> list:
        return [d.dim_value if d.dim_value > 0 else d.dim_param
                for d in value_info.type.tensor_type.shape.dim]

    print(f"\nExported: {onnx_path} ({Path(onnx_path).stat().st_size / (1024 * 1024):.1f} MB)")
    for inp in graph.input:
        print(f"  input:  {inp.name}  shape={shape_of(inp)}")
    for out in graph.output:
        print(f"  output: {out.name}  shape={shape_of(out)}")

    if args.out:
        import shutil
        args.out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, args.out / "best.onnx")
        shutil.copy2(args.weights, args.out / "best.pt")
        print(f"Copied deliverables to {args.out / 'best.onnx'} and {args.out / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
