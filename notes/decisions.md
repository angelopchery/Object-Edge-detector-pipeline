# Decisions log

Append-only record of non-obvious choices: what was decided, alternatives,
why. Raw material for the README's Assumptions section and the live round.

## 2026-09-01 — Environment

- **Replaced the Python 3.14 venv with 3.11.9** rather than trying 3.14:
  CUDA torch and onnxruntime wheel coverage for 3.14 is uncertain; 3.11 is
  the safest widely-supported floor. Alternative: 3.12 (kept as fallback if
  any wheel is missing).
- **Machine has a stale pip config** (`C:\ProgramData\pip\pip.ini` and the
  user-level pip.ini) adding the retired `pypi.ngc.nvidia.com` extra index,
  which no longer resolves and made every install retry against dead DNS.
  Fixed with a venv-level `artikate/pip.ini` override (index-url pypi.org,
  empty extra-index-url) instead of editing machine config — least invasive,
  reproducible from the README.

## 2026-09-01 — Scene identity

- **Scene identity comes from visual clustering (pHash + single-linkage),
  not filenames or timestamps.** Filenames are camera originals with no
  scene encoding; timestamp clustering leaves an 87-image continuous burst
  that cannot be split by time. Hand-partitioning 87 near-continuous frames
  is not reliable either. The clustering threshold is chosen from an
  auditable scan (notes/scene_clustering.md) and the actual guarantee is the
  cross-split leakage audit (notes/leakage_report.md), not the clustering.
- **split_dataset accepts a scene map marked `validated_by: leakage-audit`**
  (from cluster_scenes.py) as an alternative to human `reviewed: true` —
  the audit is a stronger, reproducible guarantee than eyeballing.

## 2026-09-01 — Split shape

- **Greedy-balanced split (--test-frac 0.33, --val-frac 0.2)**: whole scenes
  assigned largest-first to whichever of test/val has the biggest image
  deficit, rest to train. Alternative was pure seeded shuffling of scene IDs,
  which on ~20-40 unevenly sized clusters can produce badly skewed split
  sizes. Class balance is checked after the fact and reported, never fixed
  by resampling (PLAN Phase 3 rule).
