"""Static INT8 quantisation of the ONNX model with ONNX Runtime.

Calibration images are drawn ONLY from the training split. Using validation
(or test) images for calibration would leak evaluation data into the model:
the activation ranges would be tuned on the same images later used to score
accuracy, making the INT8 accuracy numbers optimistic.

Also supports an FP16 conversion as a fallback if INT8 accuracy drops too far.

Usage:
    python scripts/quantize.py --onnx runs/detect/train/weights/best.onnx \
        --calib-images data/dataset/images/train
    python scripts/quantize.py --onnx runs/detect/train/weights/best.onnx --fp16
"""

import argparse
import random
import sys
from pathlib import Path

from common import list_images, preprocess


class TrainSplitCalibrationReader:
    """CalibrationDataReader feeding letterboxed TRAIN-split images.

    Train split only — never val or test (see module docstring: leak).
    """

    def __init__(self, image_paths: list[Path], input_name: str, imgsz: int):
        self.image_paths = image_paths
        self.input_name = input_name
        self.imgsz = imgsz
        self._iter = iter(image_paths)

    def get_next(self):
        path = next(self._iter, None)
        if path is None:
            return None
        tensor, _, _, _ = preprocess(path, self.imgsz)
        return {self.input_name: tensor}


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description="INT8 static quantisation (train-split calibration) or FP16 conversion.")
    parser.add_argument("--onnx", type=Path, required=True, help="FP32 ONNX model")
    parser.add_argument("--calib-images", type=Path, help="TRAIN split images folder (required for INT8)")
    parser.add_argument("--num-calib", type=int, default=64, help="Calibration images to use (default 64)")
    parser.add_argument("--imgsz", type=int, default=640, help="Model input size (default 640)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for calibration image sampling")
    parser.add_argument("--fp16", action="store_true", help="Do FP16 conversion instead of INT8")
    parser.add_argument("--out", type=Path, help="Output path (default: <onnx stem>_int8.onnx / _fp16.onnx)")
    args = parser.parse_args()

    if not args.onnx.is_file():
        print(f"ERROR: ONNX model not found: {args.onnx}", file=sys.stderr)
        return 1

    if args.fp16:
        import onnx
        from onnxconverter_common import float16

        out_path = args.out or args.onnx.with_name(args.onnx.stem + "_fp16.onnx")
        model = onnx.load(str(args.onnx))
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, str(out_path))
    else:
        if not args.calib_images or not args.calib_images.is_dir():
            print("ERROR: --calib-images (the TRAIN split folder) is required for INT8", file=sys.stderr)
            return 1
        if "val" in str(args.calib_images).lower() or "test" in str(args.calib_images).lower():
            print(f"ERROR: refusing to calibrate on '{args.calib_images}' — calibration must use "
                  "the train split only, otherwise evaluation data leaks into the model.",
                  file=sys.stderr)
            return 1

        import onnxruntime as ort
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

        images = list_images(args.calib_images)
        if len(images) < args.num_calib:
            print(f"NOTE: only {len(images)} train images available, using all of them")
        random.Random(args.seed).shuffle(images)
        calib_images = images[: args.num_calib]

        input_name = ort.InferenceSession(
            str(args.onnx), providers=["CPUExecutionProvider"]
        ).get_inputs()[0].name
        reader = TrainSplitCalibrationReader(calib_images, input_name, args.imgsz)

        out_path = args.out or args.onnx.with_name(args.onnx.stem + "_int8.onnx")
        print(f"Calibrating on {len(calib_images)} train-split images (seed {args.seed})...")
        quantize_static(
            model_input=str(args.onnx),
            model_output=str(out_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
        )

    print(f"\nFP32 model:      {args.onnx}  ({size_mb(args.onnx):.2f} MB)")
    print(f"Quantised model: {out_path}  ({size_mb(out_path):.2f} MB)")
    print(f"Size reduction:  {(1 - size_mb(out_path) / size_mb(args.onnx)) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
