# Part A4 — Failure analysis (validation split, FP32 ONNX)

Produced from `scripts/render_predictions.py --onnx models/best.onnx --split val`
(green = ground truth, red = prediction; images in `notes/failure_examples/`).
Val at conf 0.25: 2 missed GT boxes and 4 false positives across 21 images.
The three worst images share one thread: **every one involves the white
earphone case.**

## 1. `err02_IMG_20260901_150211624_MP.jpg` — class confusion

- **Predicted:** `charger_brick` 0.60, box tightly on the object.
- **Should have been:** `earphone_case` (the white case on the glass table).
- **Hypothesis (specific):** the dataset contains two case units, black and
  white, and two charger units, black and white. The white case shares
  colour, glossiness, and rounded-rectangle geometry with the white charger,
  and the scene-level split concentrated the white-case-on-glass-table
  scenes in val/test (the Phase 4 leakage re-audit flagged exactly these
  frames as visually closest to train images *of the white charger* — same
  table, same palette). Localisation is perfect; only the class flips: the
  model learned "white rounded object on glass = charger".
- **Fix:** capture more white-case examples across backgrounds for train
  (the cheapest fix is moving one white-case scene from test to train and
  re-splitting — not done, because the test set was frozen); longer term,
  the D3 loop treats exactly this as a flagged-frame retraining case.

## 2. `err01_IMG_20260901_145627625_HDR.jpg` — same-colour adjacency miss

- **Predicted:** `charger_brick` 0.87 only.
- **Should have been:** both objects — the white case sits directly behind
  the white charger, overlapping.
- **Hypothesis (specific):** white-on-white adjacency. The training split
  contains only 8 overlapping-pair images total (annotation-evidence audit),
  and the white-case appearance is itself under-represented in train; the
  two factors compound. The predicted charger box is also dragged upward
  toward the case — the model sees one white blob.
- **Fix:** targeted captures of the two same-colour objects touching, at
  several angles; verify with the boxes-rendered contact sheet.

## 3. `err01_IMG_20260901_150203196_HDR.jpg` — cross-class duplicate

- **Predicted:** `earphone_case` (correct, matches GT) **plus** a second
  overlapping `charger_brick` box on the same object.
- **Should have been:** the single case box.
- **Hypothesis (specific):** the same white-case/white-charger ambiguity at
  lower confidence, surviving because NMS is per-class by design — a
  cross-class duplicate cannot be suppressed (ANSWERS.md B3.4 explains why
  per-class NMS is still correct; the fix belongs in the model's
  discrimination, not in class-agnostic NMS, which would delete correct
  boxes when the two objects genuinely overlap, as in failure 2).
- **Fix:** same as failure 1 — the confusion, not the NMS, is the defect.

## Connection to measured dataset facts

- Two physical units per class; the white case is the failure locus. The
  scene split (correctly) kept white-case-on-glass scenes out of train, so
  the model met that appearance thinly at train time. This is the honest
  cost of a leak-free split on a tiny dataset: val/test genuinely contain
  appearances train lacks.
- Single 14-minute capture session; brightness std 21.8 (one lighting
  regime) — none of the top failures are lighting-driven *within* this
  dataset, because there is no lighting variation to fail on. The
  variation report predicts lighting failures will appear first in any
  deployment outside this room.
- Per-class val→test drop concentrates in `earphone_case` (0.915→0.779
  AP50 custom evaluator) — consistent with the white-case hypothesis, since
  test holds 28 case instances across more white-case scenes.
- 95 vs 113 instance imbalance is mild and does not explain the asymmetry;
  the appearance split does.

## Addendum — live test on 8 never-seen evening photos (post-evaluation)

Provenance: these frames were shot at 19:39 (different lighting from all
training data) AFTER every metric above was frozen; nothing was tuned on
them. `scripts/detect_folder.py` results: correct detections on 7/8.

- **New failure mode: scale extrapolation.** The one complete miss is the
  white charger in extreme close-up, filling well over half the frame. The
  largest training box was 34% of frame — the model never saw this scale.
  Distinct from every val/test failure (which are appearance-driven), and
  the fix is capture, not architecture: a handful of very-close frames.
- The known white-case/white-charger confusion reproduced exactly (a
  charger_brick 0.57 duplicate on a correctly-detected 0.80 case), plus one
  weak 0.28 false positive on a laptop corner in the dimmer lighting —
  consistent with the variation report's prediction that lighting outside
  the single training regime is the first thing to degrade.
