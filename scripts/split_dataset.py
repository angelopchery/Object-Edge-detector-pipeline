"""Split the dataset into train/val/test BY SCENE, never by image.

Images within one scene (scene01_a.jpg, scene01_b.jpg, ...) are near-duplicates
of the same physical arrangement. If two frames from the same scene landed in
different splits, validation/test metrics would be inflated by memorisation
rather than measuring generalisation. So the unit of splitting is the scene:
every image from a scene goes to exactly one split.

The split is seeded and reproducible, and the full assignment is written to
data/split_manifest.json, which is committed so the split is auditable.

Usage:
    python scripts/split_dataset.py --images data/resized --labels data/labels_all
    python scripts/split_dataset.py --images data/resized --labels data/labels_all --seed 42 --train 0.70 --val 0.15 --test 0.15
"""

import argparse
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCENE_RE = re.compile(r"scene(\d+)", re.IGNORECASE)
# Measured from the makesense export (coordinate match vs CSV): 0=EarphoneCase, 1=ChargingCase.
CLASS_NAMES = {0: "earphone_case", 1: "charger_brick"}


def scene_id(path: Path) -> str:
    """Extract the scene ID from a filename like scene01_a.jpg -> 'scene01'."""
    m = SCENE_RE.search(path.stem)
    if m is None:
        print(f"ERROR: filename does not contain a scene ID: {path.name}", file=sys.stderr)
        sys.exit(1)
    return f"scene{int(m.group(1)):02d}"


def count_instances(label_path: Path) -> dict[int, int]:
    """Count boxes per class in one YOLO label file (missing file = 0 boxes)."""
    counts: dict[int, int] = defaultdict(int)
    if not label_path.is_file():
        return counts
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if line:
            counts[int(line.split()[0])] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scene-level train/val/test split. Writes data/split_manifest.json."
    )
    parser.add_argument("--images", type=Path, required=True, help="Folder with all resized images")
    parser.add_argument("--labels", type=Path, required=True, help="Folder with all YOLO label .txt files")
    parser.add_argument("--out", type=Path, default=Path("data/dataset"), help="Output dataset root (default data/dataset)")
    parser.add_argument("--manifest", type=Path, default=Path("data/split_manifest.json"), help="Manifest output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--train", type=float, default=0.70, help="Train fraction of scenes (default 0.70)")
    parser.add_argument("--val", type=float, default=0.15, help="Val fraction of scenes (default 0.15)")
    parser.add_argument("--test", type=float, default=0.15, help="Test fraction of scenes (default 0.15)")
    parser.add_argument("--test-scenes", type=Path, default=None,
                        help="JSON file listing scene IDs reserved as the held-out test set "
                             "(e.g. data/heldout_scenes.json). When given, exactly those scenes "
                             "become the test split and the rest are split train/val.")
    parser.add_argument("--scene-map", type=Path, default=None,
                        help="JSON mapping scene IDs to image filenames (from cluster_scenes.py "
                             "or build_scene_map.py), for datasets whose filenames do not encode "
                             "the scene. Accepted when marked reviewed:true or "
                             "validated_by:leakage-audit.")
    parser.add_argument("--test-frac", type=float, default=None,
                        help="Greedy-balanced mode: fraction of images for test (e.g. 0.33). "
                             "Overrides the scene-count --train/--val/--test fractions.")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Greedy-balanced mode: fraction of the NON-test images for val "
                             "(default 0.2; only used with --test-frac)")
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        print("ERROR: --train + --val + --test must sum to 1.0", file=sys.stderr)
        return 1
    if not args.images.is_dir():
        print(f"ERROR: images folder not found: {args.images}", file=sys.stderr)
        return 1
    if not args.labels.is_dir():
        print(f"ERROR: labels folder not found: {args.labels}", file=sys.stderr)
        return 1

    # Group images by scene: either from a reviewed scene map (camera-named
    # files) or from the sceneNN filename convention.
    name_to_scene: dict[str, str] = {}
    if args.scene_map:
        loaded = json.loads(args.scene_map.read_text())
        # Two acceptable provenances: a human-reviewed map, or a visual
        # clustering whose split is gated by the leakage audit (Phase 3c).
        if not (loaded.get("reviewed", False) or loaded.get("validated_by") == "leakage-audit"):
            print(f"ERROR: {args.scene_map} is neither reviewed:true nor "
                  "validated_by:leakage-audit. An unvetted map can silently leak "
                  "near-duplicates across splits.", file=sys.stderr)
            return 1
        for sid, names in loaded["scenes"].items():
            for n in names:
                name_to_scene[n] = sid

    scenes: dict[str, list[Path]] = defaultdict(list)
    unmapped: list[str] = []
    for img in sorted(args.images.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        if args.scene_map:
            sid = name_to_scene.get(img.name)
            if sid is None:
                unmapped.append(img.name)
                continue
            scenes[sid].append(img)
        else:
            scenes[scene_id(img)].append(img)

    if unmapped:
        print(f"ERROR: {len(unmapped)} images missing from the scene map: {unmapped[:5]} ...",
              file=sys.stderr)
        return 1
    if not scenes:
        print(f"ERROR: no images found in {args.images}", file=sys.stderr)
        return 1

    # Optional pre-reserved held-out test scenes (PLAN stage 2): those scenes
    # become the test split outright and are excluded from the shuffle.
    forced_test: list[str] = []
    if args.test_scenes:
        loaded = json.loads(args.test_scenes.read_text())
        forced_test = loaded["test_scenes"] if isinstance(loaded, dict) else loaded
        unknown = sorted(set(forced_test) - set(scenes))
        if unknown:
            print(f"ERROR: held-out scenes not found among images: {unknown}", file=sys.stderr)
            return 1

    # Shuffle SCENE IDs (not images) with a fixed seed, then cut by fractions.
    scene_ids = sorted(set(scenes) - set(forced_test))
    rng = random.Random(args.seed)
    rng.shuffle(scene_ids)

    n = len(scene_ids)
    if args.test_frac is not None:
        # Greedy-balanced mode (PLAN Phase 3b): assign whole scenes, largest
        # first, to whichever of test/val still has the largest image deficit
        # against its target; everything else goes to train. Shuffle first so
        # equal-sized scenes break ties by seed, not by name.
        total_images = sum(len(v) for v in scenes.values())
        test_target = args.test_frac * total_images
        val_target = args.val_frac * (total_images - test_target)
        assign: dict[str, list[str]] = {"train": [], "val": [], "test": list(forced_test)}
        counts = {"train": 0, "val": 0,
                  "test": sum(len(scenes[s]) for s in forced_test)}
        for sid in sorted(scene_ids, key=lambda s: (-len(scenes[s]), scene_ids.index(s))):
            deficits = {"test": test_target - counts["test"], "val": val_target - counts["val"]}
            split = max(deficits, key=deficits.get) if max(deficits.values()) > 0 else "train"
            assign[split].append(sid)
            counts[split] += len(scenes[sid])
        assignment = {k: sorted(v) for k, v in assign.items()}
    elif forced_test:
        # Test is fixed by the manifest; split the remainder train/val
        # by the train:val ratio.
        n_train = round(n * args.train / (args.train + args.val))
        assignment = {
            "train": sorted(scene_ids[:n_train]),
            "val": sorted(scene_ids[n_train:]),
            "test": sorted(forced_test),
        }
    else:
        n_train = round(n * args.train)
        n_val = round(n * args.val)
        assignment = {
            "train": sorted(scene_ids[:n_train]),
            "val": sorted(scene_ids[n_train:n_train + n_val]),
            "test": sorted(scene_ids[n_train + n_val:]),
        }

    # Hard guarantee: scene sets must be pairwise disjoint.
    train_s, val_s, test_s = (set(assignment[s]) for s in ("train", "val", "test"))
    assert not (train_s & val_s), f"train/val scene overlap: {train_s & val_s}"
    assert not (train_s & test_s), f"train/test scene overlap: {train_s & test_s}"
    assert not (val_s & test_s), f"val/test scene overlap: {val_s & test_s}"
    assert train_s | val_s | test_s == set(scenes), "some scenes were not assigned to any split"

    # Copy files into the dataset layout and gather stats.
    manifest = {
        "seed": args.seed,
        "mode": ("greedy-balanced" if args.test_frac is not None else "scene-count-fractions"),
        "fractions": ({"test_frac": args.test_frac, "val_frac": args.val_frac}
                      if args.test_frac is not None
                      else {"train": args.train, "val": args.val, "test": args.test}),
        "test_scenes_file": str(args.test_scenes) if args.test_scenes else None,
        "scene_map_file": str(args.scene_map) if args.scene_map else None,
        "scenes": assignment,
        "images": {},
    }
    print(f"{len(scenes)} scenes, {sum(len(v) for v in scenes.values())} images, seed={args.seed}"
          + (f" ({len(forced_test)} scenes pre-reserved for test)" if forced_test else "") + "\n")

    for split in ("train", "val", "test"):
        img_dir = args.out / "images" / split
        lbl_dir = args.out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        split_images: list[str] = []
        class_counts: dict[int, int] = defaultdict(int)
        missing_labels: list[str] = []

        for sid in assignment[split]:
            for img in scenes[sid]:
                shutil.copy2(img, img_dir / img.name)
                lbl = args.labels / (img.stem + ".txt")
                if lbl.is_file():
                    shutil.copy2(lbl, lbl_dir / lbl.name)
                else:
                    missing_labels.append(img.name)
                for cls, cnt in count_instances(lbl).items():
                    class_counts[cls] += cnt
                split_images.append(img.name)

        manifest["images"][split] = split_images
        counts_str = ", ".join(
            f"{CLASS_NAMES.get(c, c)}={class_counts.get(c, 0)}" for c in sorted(CLASS_NAMES)
        )
        print(f"{split:5s}: {len(assignment[split]):2d} scenes ({', '.join(assignment[split])})")
        print(f"       {len(split_images)} images | instances: {counts_str}")
        if missing_labels:
            print(f"       WARNING: {len(missing_labels)} images have no label file: {missing_labels}")
        print()

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Split manifest written to {args.manifest} — commit this file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
