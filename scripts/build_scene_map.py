"""Draft a scene map by clustering camera-filename timestamps.

The dataset filenames are camera originals (IMG_YYYYMMDD_HHMMSSmmm_*.jpg),
not the sceneNN_x.jpg convention the scene split assumed. This script groups
consecutive shots separated by less than --gap seconds into one scene and
writes data/scene_map.json.

THE OUTPUT IS A DRAFT. Timestamp gaps cannot distinguish "walked to a new
arrangement quickly" from "kept shooting the same one" — the map must be
reviewed by the person who took the photos, scene boundaries corrected by
editing the JSON, and only then committed. The split is only as honest as
this file, so review it seriously: two near-identical arrangements in two
"scenes" is a leak the split cannot catch.

Usage:
    python scripts/build_scene_map.py --images YoloData --gap 15
    # review + edit data/scene_map.json, set "reviewed": true, commit
"""

import argparse
import json
import re
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
TS_RE = re.compile(r"IMG_(\d{8})_(\d{2})(\d{2})(\d{2})(\d{3})")


def timestamp_ms(name: str) -> int:
    m = TS_RE.match(name)
    if m is None:
        print(f"ERROR: cannot parse timestamp from filename: {name}", file=sys.stderr)
        sys.exit(1)
    hh, mm, ss, ms = (int(m.group(i)) for i in range(2, 6))
    return ((hh * 60 + mm) * 60 + ss) * 1000 + ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Draft scene_map.json by clustering capture timestamps.")
    parser.add_argument("--images", type=Path, required=True, help="Folder of camera-named images")
    parser.add_argument("--gap", type=float, default=15.0,
                        help="Seconds of silence that starts a new scene (default 15)")
    parser.add_argument("--out", type=Path, default=Path("data/scene_map.json"), help="Output JSON")
    parser.add_argument("--warn-size", type=int, default=12,
                        help="Warn for scenes with more images than this (default 12)")
    args = parser.parse_args()

    names = sorted((p.name for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS),
                   key=timestamp_ms)
    if not names:
        print(f"ERROR: no images in {args.images}", file=sys.stderr)
        return 1

    scenes: list[list[str]] = [[names[0]]]
    for prev, cur in zip(names, names[1:]):
        if (timestamp_ms(cur) - timestamp_ms(prev)) / 1000 > args.gap:
            scenes.append([])
        scenes[-1].append(cur)

    scene_map = {f"scene{i + 1:02d}": imgs for i, imgs in enumerate(scenes)}

    print(f"{len(names)} images -> {len(scenes)} scenes at gap>{args.gap}s\n")
    oversized = []
    for sid, imgs in scene_map.items():
        span = (timestamp_ms(imgs[-1]) - timestamp_ms(imgs[0])) / 1000
        flag = "  <-- REVIEW: large scene, likely multiple arrangements" if len(imgs) > args.warn_size else ""
        print(f"  {sid}: {len(imgs):3d} images over {span:6.1f}s  ({imgs[0]} .. {imgs[-1]}){flag}")
        if len(imgs) > args.warn_size:
            oversized.append(sid)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"generated_by": f"build_scene_map.py --gap {args.gap}",
         "reviewed": False,
         "scenes": scene_map}, indent=2) + "\n")
    print(f"\nDraft written to {args.out} (reviewed: false).")
    if oversized:
        print(f"Scenes needing manual splitting: {', '.join(oversized)}")
    print("Edit scene boundaries by hand, set \"reviewed\": true, then commit the file.")
    print("split_dataset.py refuses a map that is not marked reviewed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
