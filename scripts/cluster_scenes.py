"""Derive scene identity by visual similarity (PLAN Phase 3a).

Filenames carry no scene IDs and timestamp clustering cannot separate an
87-image continuous burst, so scene identity is derived from image content.

Method selection was MEASURED, not assumed (see notes/decisions.md):
64-bit whole-image pHash gave no separation on this dataset (temporally
adjacent frames: mean Hamming 29.4 vs 31.5 for frames >60s apart) because
handheld close-up reframing changes global structure between consecutive
shots. Coarse HSV colour-histogram correlation separates cleanly
(adjacent median distance 0.236 vs distant 0.916), so 'colorhist' is the
default; 'phash' is kept selectable for comparison.

  1. per-image feature: 8x4x4 HSV histogram (colorhist) or 64-bit pHash
  2. single-linkage agglomerative clustering (union-find) on pairwise
     distance: images join a scene if distance <= threshold
  3. capture timestamps as a weak prior: pairs shot within --time-slack
     seconds join at a slightly looser threshold (+prior bonus)

The threshold is chosen by scanning a range and taking the knee of the
cluster-count curve (max second difference); the whole scan is written to
notes/scene_clustering.md so the choice is auditable, and --threshold N
overrides it.

The leakage audit (check_leakage.py, Phase 3c) is the actual gate on the
split — this clustering is the means, not the guarantee.

Usage:
    python scripts/cluster_scenes.py --images data/prepared --out data/scene_map.json --method colorhist --threshold auto
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from common import list_images

TS_RE = re.compile(r"IMG_(\d{8})_(\d{2})(\d{2})(\d{2})(\d{3})")


def phash(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].flatten()
    low = dct[1:]  # drop the DC coefficient
    return (low > np.median(low))


def colorhist(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, None).flatten().astype(np.float32)


def pairwise_distance(features: list[np.ndarray], method: str) -> np.ndarray:
    n = len(features)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if method == "phash":
                d = float(np.count_nonzero(features[i] != features[j]))
            else:  # colorhist: 1 - correlation, in [0, 2]
                d = 1.0 - cv2.compareHist(features[i], features[j], cv2.HISTCMP_CORREL)
            dist[i, j] = dist[j, i] = d
    return dist


def timestamp_s(name: str) -> float | None:
    m = TS_RE.match(name)
    if m is None:
        return None
    hh, mm, ss, ms = (int(m.group(i)) for i in range(2, 6))
    return ((hh * 60 + mm) * 60 + ss) + ms / 1000


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def cluster(dist: np.ndarray, times: list[float | None], thr: float,
            time_slack: float, prior_bonus: float) -> list[int]:
    n = len(dist)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            limit = thr
            if (times[i] is not None and times[j] is not None
                    and abs(times[i] - times[j]) <= time_slack):
                limit += prior_bonus
            if dist[i, j] <= limit:
                uf.union(i, j)
    roots = {}
    return [roots.setdefault(uf.find(i), len(roots)) for i in range(n)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Visual scene clustering via perceptual hash.")
    parser.add_argument("--images", type=Path, required=True, help="Images folder")
    parser.add_argument("--out", type=Path, default=Path("data/scene_map.json"), help="Scene map output")
    parser.add_argument("--method", choices=("colorhist", "phash"), default="colorhist",
                        help="Similarity method (default colorhist — measured to separate; "
                             "phash does not on this dataset)")
    parser.add_argument("--threshold", default="auto",
                        help="Distance threshold (Hamming 0-63 for phash, 1-corr 0-2 for "
                             "colorhist), or 'auto' for the knee of the scan (default)")
    parser.add_argument("--time-slack", type=float, default=5.0,
                        help="Seconds within which the timestamp prior applies (default 5)")
    parser.add_argument("--prior-bonus", type=float, default=None,
                        help="Extra distance tolerance for temporally adjacent shots "
                             "(default: 3 for phash, 0.05 for colorhist)")
    parser.add_argument("--scan-report", type=Path, default=Path("notes/scene_clustering.md"))
    args = parser.parse_args()

    images = list_images(args.images)
    if not images:
        print(f"ERROR: no images in {args.images}", file=sys.stderr)
        return 1

    feature_fn = colorhist if args.method == "colorhist" else phash
    features = [feature_fn(p) for p in images]
    dist = pairwise_distance(features, args.method)
    times = [timestamp_s(p.name) for p in images]

    if args.prior_bonus is None:
        args.prior_bonus = 3.0 if args.method == "phash" else 0.05

    # Threshold scan for the report and the auto choice.
    if args.method == "phash":
        scan_range = [float(t) for t in range(1, 26)]
    else:
        scan_range = [round(0.05 * t, 2) for t in range(1, 17)]  # 0.05 .. 0.80
    counts = [len(set(cluster(dist, times, t, args.time_slack, args.prior_bonus)))
              for t in scan_range]

    if args.threshold == "auto":
        # Knee = largest second difference in the cluster-count curve.
        second_diff = [counts[i - 1] - 2 * counts[i] + counts[i + 1]
                       for i in range(1, len(counts) - 1)]
        thr = scan_range[1 + int(np.argmax(second_diff))]
    else:
        thr = float(args.threshold)

    labels = cluster(dist, times, thr, args.time_slack, args.prior_bonus)
    n_clusters = len(set(labels))

    scenes: dict[str, list[str]] = {}
    for idx, lab in enumerate(labels):
        scenes.setdefault(f"scene{lab + 1:02d}", []).append(images[idx].name)
    # Renumber by first image so scene IDs are stable/deterministic.
    ordered = dict(sorted(scenes.items(), key=lambda kv: min(kv[1])))
    scenes = {f"scene{i + 1:02d}": sorted(v) for i, (_, v) in enumerate(ordered.items())}

    # Scan report.
    lines = [
        "# Scene clustering threshold scan",
        "",
        f"Produced by `scripts/cluster_scenes.py` (method: {args.method}, single-linkage",
        f"union-find, timestamp prior: +{args.prior_bonus} distance within {args.time_slack}s).",
        "",
        "| threshold | clusters |",
        "|-----------|----------|",
    ]
    lines += [f"| {t} | {c} |" for t, c in zip(scan_range, counts)]
    lines += ["", f"Chosen threshold: **{thr}** "
              + ("(auto: knee of the curve, max second difference)" if args.threshold == "auto"
                 else "(manual override)"),
              f"Clusters at chosen threshold: **{n_clusters}**",
              "",
              "Cluster sizes: " + str(sorted((len(v) for v in scenes.values()), reverse=True))]
    args.scan_report.parent.mkdir(parents=True, exist_ok=True)
    args.scan_report.write_text("\n".join(lines) + "\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"generated_by": f"cluster_scenes.py --method {args.method} --threshold {thr}",
         "validated_by": "leakage-audit",
         "n_clusters": n_clusters,
         "scenes": scenes}, indent=2) + "\n")

    print(f"{len(images)} images -> {n_clusters} scenes at {args.method} distance <= {thr}")
    print("cluster sizes:", sorted((len(v) for v in scenes.values()), reverse=True))
    print(f"scan: {args.scan_report}\nmap:  {args.out}")

    if n_clusters < 10:
        print("\nWARNING: under 10 clusters — effective dataset size is much smaller than "
              "150 images implies. This is a major finding for Known Gaps (PLAN Phase 3a).")
    elif n_clusters > 60:
        print("\nWARNING: over 60 clusters — threshold likely too tight, near-duplicates "
              "will be separated and the split may leak. Consider loosening (PLAN Phase 3a).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
