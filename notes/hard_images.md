# Hard images (A2)

**Provenance, stated plainly:** this list was identified **post-hoc** — by
ranking validation and test images on measured prediction error
(`scripts/render_predictions.py`, `errNN_` prefix = missed GT + false
positives) — not declared at capture time. Status: **proposed by the
error ranking, pending annotator amendment** (Phase 0 item 4).

All ten involve the failure locus from
[failure_analysis.md](failure_analysis.md): the white earphone case, alone
or adjacent to the white charger, on the glass table.

## From validation (overlays in `runs/render_val/`)

| Image | Errors | Why it is hard |
|---|---|---|
| IMG_20260901_150211624_MP | 1 miss + 1 FP | white case classified as charger_brick 0.60 — white-on-white class confusion |
| IMG_20260901_145627625_HDR | 1 miss | white case directly behind white charger; case missed entirely |
| IMG_20260901_150203196_HDR | 1 FP | cross-class duplicate: correct case box plus spurious charger box |
| IMG_20260901_150200427_HDR | 1 FP | same white-case/charger ambiguity, lower confidence |
| IMG_20260901_145629922_HDR | 1 FP | white pair adjacent, spurious extra box |

## From test (overlays in `runs/render_test/`; evaluated once, listed after)

| Image | Errors | Why it is hard |
|---|---|---|
| IMG_20260901_150213759_HDR | 1 miss + 2 FP | worst test image; white case on glass |
| IMG_20260901_145633962_HDR | 1 miss + 2 FP | white pair, edge-cropped charger (left border) |
| IMG_20260901_145619608_HDR | 3 FP | overlapping white pair, duplicate boxes both classes |
| IMG_20260901_150140864_HDR | 1 miss + 1 FP | white case, reflective glass background |
| IMG_20260901_150046375_HDR | 1 miss + 1 FP | white case, HDR frame, low contrast against table |

## What was expected to be hard vs what is

Pre-capture intuition would have named occlusion, distance, and blur. The
measured answer is narrower and more interesting: **intra-class colour
overlap between the two white units** dominates every top error; classic
occlusion appears only in combination with it. The blur observed in some
frames (e.g. the class-check crop) did not surface in the top errors.
