# Annotation Guide

> **Provenance — read this first.** All 150 images were hand-labelled in
> makesense.ai in a single pass *before* this guide was finalised (a failed
> process gate, recorded in Known Gaps). This document is therefore written
> **descriptively**: the rules below were inferred from the labels
> themselves by `scratchpad/annotation_evidence.py`-style analysis, each
> marked `[inferred]` with its evidence, and then confirmed or corrected by
> the annotator. It describes observed practice, not intent.

## Classes

| ID | makesense name | Meaning |
|----|----------------|---------|
| 0 | `EarphoneCase` | earphone case — **2 distinct physical units** in the dataset (green boxes in makesense) |
| 1 | `ChargingCase` | charger brick (wall adapter) — **2 distinct physical units**, one black and one white (red boxes in makesense) |

Confirmed by the annotator 2026-09-01 against the crops in `notes/class_check/`.

The ID→name mapping was **measured** by matching the YOLO txt export against
the CSV export coordinate-by-coordinate (208/208 boxes agree). Do not trust
the class names alone: "ChargingCase" is ambiguous between a charger brick
and a charging case, which is why a cropped example of each class is kept in
`notes/class_check/` and was confirmed by eye.

## Rules, as practised

### 1. Frame-edge crops — labelled, visible extent only `[inferred]`
Evidence: 10 boxes touch a frame border (left/right/top), with areas 4–19%
of frame — partially cut-off objects were boxed rather than skipped, and
boxes stop at the border (all coordinates within [0,1], none clamped
degenerate).

### 2. Occlusion / adjacency — labelled through `[inferred]`
Evidence: 8 images contain overlapping boxes (IoU up to 0.17), i.e. the two
objects touching or partially occluding each other; both objects are always
labelled in those frames. No frame shows an object skipped for being
partially hidden.

### 3. At most one instance of each class per frame `[inferred, corrected by annotator]`
Evidence: 0 images contain two boxes of the same class, across all 150.
Original inference ("one physical unit per class") was **wrong**: the
annotator confirms **four physical objects** — two charger bricks (one
black, one white) and two earphone cases — with never more than one unit
of a class in frame. Consequence for the README: intra-class appearance
variation exists (notably black vs white charger), but generalisation
beyond these four specific units is still unmeasured.

### 4. Box tightness — body only, no cable, no shadow `[confirmed 2026-09-01]`
Boxes are tight to the object silhouette (aspect ratios track the objects'
physical proportions: median w/h 1.39 for the case, 1.15 for the brick,
varying with pose 0.6–2.5). Annotator confirms: charger boxes cover the
brick body/prongs only — cables and cast shadows excluded.

### 5. Minimum instance size — none needed `[inferred]`
Evidence: smallest box is 1.69% of frame (~85×110 px at 1280 long edge).
No tiny/distant instances exist, so no minimum-size rule was ever exercised.
The 0.1%-of-frame suspicion threshold in `verify_labels.py` never fired.

### 6. Background instances — none exist `[confirmed 2026-09-01]`
Annotator confirms no frame contains a second charger or case visible in
the background that was left unlabelled. Every visible instance is boxed.

### 7. Open earphone case — never occurs `[confirmed 2026-09-01]`
Annotator confirms the case is closed in every frame, so the
one-box-or-two question never arose. If open-case frames are ever added,
the rule must be decided before labelling them.

## Process record

- Tool: makesense.ai; exports: YOLO txt (authoritative), VOC XML and CSV
  (provenance only — their absolute pixel coordinates go stale after the
  1280px resize; YOLO txt normalised coordinates survive it).
- Workflow: all 150 images hand-labelled in one session. No seed model, no
  model-assisted pre-labelling was used (contrary to the original plan —
  `prelabel.py` exists but was never part of this dataset's history).
- Verification: `verify_labels.py` — 0 hard errors (no orphans, all
  coordinates in [0,1], no degenerate boxes, class IDs only {0,1}).
- Cross-format check: YOLO txt vs CSV vs VOC XML agree box-for-box (208).
