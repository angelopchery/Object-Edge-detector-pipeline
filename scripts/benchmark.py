"""Latency benchmark: FP32 vs quantised ONNX, same hardware, same input.

Batch size 1. 20 warmup iterations (discarded), 200 timed iterations.
Reports mean / median / p95 / std in ms, plus the ONNX Runtime execution
provider and thread count, and prints a markdown table for the README.

Usage:
    python scripts/benchmark.py --fp32 runs/detect/train/weights/best.onnx \
        --quant runs/detect/train/weights/best_int8.onnx \
        --image data/dataset/images/val/scene05_a.jpg
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common import preprocess


def bench_model(model_path: Path, tensor: np.ndarray, warmup: int, iters: int,
                threads: int | None) -> dict:
    opts = ort.SessionOptions()
    if threads is not None:
        opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(model_path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    feed = {input_name: tensor}

    for _ in range(warmup):
        sess.run(None, feed)

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms.sort()
    return {
        "name": model_path.name,
        "provider": sess.get_providers()[0],
        "threads": opts.intra_op_num_threads,  # 0 = ORT default (all cores)
        "mean": statistics.mean(times_ms),
        "median": statistics.median(times_ms),
        "p95": times_ms[int(len(times_ms) * 0.95) - 1],
        "std": statistics.stdev(times_ms),
        "size_mb": model_path.stat().st_size / (1024 * 1024),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark FP32 vs quantised ONNX latency (batch 1, CPU).")
    parser.add_argument("--fp32", type=Path, required=True, help="FP32 ONNX model")
    parser.add_argument("--quant", type=Path, required=True, help="Quantised ONNX model (INT8 or FP16)")
    parser.add_argument("--image", type=Path, required=True, help="Real image to use as input")
    parser.add_argument("--imgsz", type=int, default=640, help="Model input size (default 640)")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations (default 20)")
    parser.add_argument("--iters", type=int, default=200, help="Timed iterations (default 200)")
    parser.add_argument("--threads", type=int, default=None,
                        help="intra-op threads (default: ORT default = all physical cores)")
    args = parser.parse_args()

    for p in (args.fp32, args.quant, args.image):
        if not p.is_file():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            return 1

    tensor, _, _, _ = preprocess(args.image, args.imgsz)

    results = [
        bench_model(args.fp32, tensor, args.warmup, args.iters, args.threads),
        bench_model(args.quant, tensor, args.warmup, args.iters, args.threads),
    ]

    print(f"\nonnxruntime {ort.__version__} | provider: {results[0]['provider']} | "
          f"intra-op threads: {results[0]['threads'] or 'ORT default'} | "
          f"batch 1, {args.warmup} warmup + {args.iters} timed iterations\n")

    print("| Model | Size (MB) | Mean (ms) | Median (ms) | p95 (ms) | Std (ms) |")
    print("|-------|-----------|-----------|-------------|----------|----------|")
    for r in results:
        print(f"| {r['name']} | {r['size_mb']:.2f} | {r['mean']:.2f} | "
              f"{r['median']:.2f} | {r['p95']:.2f} | {r['std']:.2f} |")

    speedup = results[0]["mean"] / results[1]["mean"]
    print(f"\nQuantised model mean-latency speedup vs FP32: {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
