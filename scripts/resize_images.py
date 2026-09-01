"""Resize images so the long edge is at most 1280px, preserving aspect ratio.

EXIF orientation is applied (baked in) before resizing so that annotation
tools, training, and inference all see the same pixel layout. Originals are
never touched: output always goes to a separate folder.

Usage:
    python scripts/resize_images.py --src data/raw --dst data/resized
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def folder_stats(folder: Path) -> tuple[int, float]:
    """Return (file count, total size in MB) for images in a folder."""
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    total_mb = sum(p.stat().st_size for p in files) / (1024 * 1024)
    return len(files), total_mb


def resize_one(src_path: Path, dst_path: Path, long_edge: int, quality: int) -> None:
    with Image.open(src_path) as im:
        # Bake in EXIF orientation so width/height match what viewers show.
        im = ImageOps.exif_transpose(im)

        w, h = im.size
        scale = long_edge / max(w, h)
        if scale < 1.0:
            new_size = (round(w * scale), round(h * scale))
            im = im.resize(new_size, Image.LANCZOS)
        # If the image is already smaller, keep it as-is (no upscaling).

        if dst_path.suffix.lower() in {".jpg", ".jpeg"}:
            im = im.convert("RGB")
            im.save(dst_path, quality=quality)
        else:
            im.save(dst_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resize images (long edge -> --long-edge px) into a new folder."
    )
    parser.add_argument("--src", type=Path, required=True, help="Source folder with originals")
    parser.add_argument("--dst", type=Path, required=True, help="Output folder (created if missing)")
    parser.add_argument("--long-edge", type=int, default=1280, help="Target long edge in px (default 1280)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    parser.add_argument("--apply-exif", action="store_true",
                        help="EXIF orientation is ALWAYS baked in; flag kept for command compatibility")
    args = parser.parse_args()

    if not args.src.is_dir():
        print(f"ERROR: source folder not found: {args.src}", file=sys.stderr)
        return 1
    if args.src.resolve() == args.dst.resolve():
        print("ERROR: --dst must be different from --src (originals are never overwritten)", file=sys.stderr)
        return 1

    args.dst.mkdir(parents=True, exist_ok=True)

    src_count, src_mb = folder_stats(args.src)
    print(f"Source: {src_count} images, {src_mb:.1f} MB total")

    processed = 0
    for src_path in sorted(args.src.iterdir()):
        if src_path.suffix.lower() not in IMAGE_EXTS:
            continue
        dst_path = args.dst / src_path.name
        resize_one(src_path, dst_path, args.long_edge, args.quality)
        processed += 1

    dst_count, dst_mb = folder_stats(args.dst)
    print(f"Processed: {processed} images")
    print(f"Output: {dst_count} images, {dst_mb:.1f} MB total ({args.dst})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
