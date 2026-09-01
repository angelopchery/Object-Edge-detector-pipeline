# Annotation Guide

> **Status: DRAFT.** These are placeholder rules — I will finalise each one
> before annotating the first image, and every box in the dataset follows the
> finalised version. If a rule changes mid-annotation, all earlier labels get
> re-reviewed against the new rule.

Classes:

| ID | Name | Definition |
|----|------|------------|
| 0 | `charger_brick` | TODO: define exactly (wall adapter only? cable attached counts as part of the box or not?) |
| 1 | `earphone_case` | TODO: define exactly (case only, or case with earphones visible inside?) |

## Rules

### 1. Occlusion threshold
**Placeholder:** label an object if at least ~20% of it is visible AND it is
identifiable as its class from this frame alone. Box the *visible extent only*
— do not guess the hidden extent (amodal boxes are not used).
TODO: confirm threshold after looking at the actual occluded frames.

### 2. Frame-edge crops
**Placeholder:** label objects cut off by the frame edge if the visible part
passes the occlusion rule above. The box goes to the image border, never
beyond it (coordinates clamp to [0, 1]).
TODO: decide a minimum visible fraction for edge cases (e.g. discard if <10% visible).

### 3. Open earphone case: one box or two?
**Placeholder:** an open case (lid + base connected by hinge) is **one box**
covering both halves, because it is one physical object. If the two halves are
physically separated in a frame, that frame goes to `notes/hard_images.md` and
the rule gets decided there.
TODO: confirm after checking whether any capture scene has a fully detached lid.

### 4. Minimum box size
**Placeholder:** do not label instances smaller than ~12px on the shortest
side at 1280px long edge (roughly matches `verify_labels.py`'s 0.1%-of-frame
suspicion threshold). Smaller than that, the object is unlearnable at
imgsz=640 anyway.
TODO: confirm the pixel threshold against real distant instances.

### 5. Out-of-focus background instances
**Placeholder:** label a blurred background instance **only if** it is still
identifiable as its class by a human who has not seen the other frames of the
scene. If identification requires scene context, leave it unlabelled and log
the image in `notes/hard_images.md`.
TODO: revisit — leaving true instances unlabelled teaches the model they are
background; if there are many, consider excluding those images entirely instead.

### 6. Box tightness
Boxes are tight to the visible pixels of the object: no padding, no clipping.
Cables/straps attached to an object: TODO decide (default: exclude cable from
`charger_brick` box; include only the brick body and prongs).

## Process discipline

- Tool: makesense.ai, YOLO txt export, class order `charger_brick=0, earphone_case=1`
  (must match `data/data.yaml`).
- ~40 images hand-labelled first → seed model → `prelabel.py` on the rest →
  **every** pre-labelled box manually reviewed and corrected. No box enters the
  dataset unreviewed.
- After every annotation batch: run `python scripts/verify_labels.py` and fix
  everything it reports before committing labels.
- Ambiguous/hard frames are logged in `notes/hard_images.md` with the decision
  taken, so the rules stay consistent.
