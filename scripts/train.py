"""Train YOLO11n on the charger_brick / earphone_case dataset.

Defaults are sized for a 4 GB RTX 3050 Laptop GPU: imgsz 640, batch 8,
AMP on. Everything is overridable from the CLI. The fully resolved training
config (what Ultralytics actually ran, not what was intended) is dumped to
resolved_config.json inside the run directory so the README can quote it.

Usage:
    python scripts/train.py
    python scripts/train.py --model yolo11n.pt --epochs 100 --batch 8 --imgsz 640 --name seed_run
"""

import argparse
import json
import sys
import time
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO11n (defaults tuned for 4 GB VRAM).")
    parser.add_argument("--model", default="yolo11n.pt", help="Base weights (default yolo11n.pt)")
    parser.add_argument("--data", default="data/data.yaml", help="Dataset yaml (default data/data.yaml)")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size (default 640)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default 8 for 4 GB VRAM)")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (default 100)")
    parser.add_argument("--patience", type=int, default=25, help="Early-stopping patience (default 25)")
    parser.add_argument("--device", default="0", help="CUDA device or 'cpu' (default 0)")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers (default 4)")
    parser.add_argument("--seed", type=int, default=42, help="Training seed (default 42)")
    parser.add_argument("--name", default="train", help="Run name under runs/detect/")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--resume", action="store_true", help="Resume the run with this name")
    args = parser.parse_args()

    model = YOLO(args.model)

    start = time.perf_counter()
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        name=args.name,
        amp=not args.no_amp,
        resume=args.resume,
        project="runs/detect",
    )
    elapsed = time.perf_counter() - start

    # Dump the config Ultralytics actually resolved and ran with.
    save_dir = Path(model.trainer.save_dir)
    resolved = {k: str(v) if isinstance(v, Path) else v for k, v in vars(model.trainer.args).items()}
    config_path = save_dir / "resolved_config.json"
    config_path.write_text(json.dumps(resolved, indent=2, default=str) + "\n")

    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nWall-clock training time: {int(hours)}h {int(minutes)}m {int(seconds)}s ({elapsed:.0f}s)")
    print(f"Run directory: {save_dir}")
    print(f"Best weights:  {save_dir / 'weights' / 'best.pt'}")
    print(f"Resolved config: {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
