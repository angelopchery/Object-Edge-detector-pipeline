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
| 0 | `EarphoneCase` | earphone case |
| 1 | `ChargingCase` | charger brick (wall adapter) — **pending visual confirmation, see notes/class_check/** |

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

### 3. Exactly one instance of each class per frame `[inferred]`
Evidence: 0 images contain two boxes of the same class, across all 150.
Interpretation: **one physical charger brick and one physical earphone case
were photographed throughout.** Consequence, stated honestly in the README:
the model learns these two specific units; validation figures overstate
generalisation to other units of the same classes.

### 4. Box tightness `[inferred, needs annotator confirmation]`
Boxes appear tight to the object silhouette (aspect ratios track the
objects' physical proportions: median w/h 1.39 for the case, 1.15 for the
brick, varying with pose 0.6–2.5). Whether the charger's cable/prongs or
cast shadows were included cannot be determined from coordinates alone —
**annotator to confirm**: TODO.

### 5. Minimum instance size — none needed `[inferred]`
Evidence: smallest box is 1.69% of frame (~85×110 px at 1280 long edge).
No tiny/distant instances exist, so no minimum-size rule was ever exercised.
The 0.1%-of-frame suspicion threshold in `verify_labels.py` never fired.

### 6. Out-of-focus / background instances — no evidence either way
No labelled box is implausibly small or peripheral, and there is no record
of skipped background instances. If any frame contains an unlabelled
background instance, it is an unknown — **annotator to confirm whether any
frames contained a second, unlabelled unit in the background**: TODO.

### 7. Open earphone case — one box or two?
Not determinable from coordinates. **Annotator to confirm** whether any
frames show the case open, and if so whether it was boxed as one object:
TODO.

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
