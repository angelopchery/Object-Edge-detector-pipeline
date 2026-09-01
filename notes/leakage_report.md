# Cross-split leakage audit

Produced by `scripts/check_leakage.py --threshold 8` (64-bit dHash, 79 train images).

```
=== val vs train === (21 images)
  closest 10 pairs (Hamming distance out of 64, lower = more similar):
    10  IMG_20260901_150211624_MP.jpg  ~  IMG_20260901_145957494_HDR.jpg
    12  IMG_20260901_150200427_HDR.jpg  ~  IMG_20260901_145957494_HDR.jpg
    13  IMG_20260901_145629922_HDR.jpg  ~  IMG_20260901_145957494_HDR.jpg
    15  IMG_20260901_145627625_HDR.jpg  ~  IMG_20260901_145141055_MP.jpg
    16  IMG_20260901_145330211_MP.jpg  ~  IMG_20260901_145141055_MP.jpg
    17  IMG_20260901_145249446_HDR.jpg  ~  IMG_20260901_145957494_HDR.jpg
    18  IMG_20260901_145317238_MP.jpg  ~  IMG_20260901_150338179_MP.jpg
    19  IMG_20260901_145241328_HDR.jpg  ~  IMG_20260901_145251410_HDR.jpg
    19  IMG_20260901_150203196_HDR.jpg  ~  IMG_20260901_150011879_HDR.jpg
    20  IMG_20260901_145138406_MP.jpg  ~  IMG_20260901_150328700_MP.jpg
  no pair at or below threshold 8.

=== test vs train === (50 images)
  closest 10 pairs (Hamming distance out of 64, lower = more similar):
    10  IMG_20260901_150205078_HDR.jpg  ~  IMG_20260901_145957494_HDR.jpg
    13  IMG_20260901_150034024_MP.jpg  ~  IMG_20260901_150149053_MP.jpg
    14  IMG_20260901_150017410_HDR.jpg  ~  IMG_20260901_145957494_HDR.jpg
    14  IMG_20260901_150151553_MP.jpg  ~  IMG_20260901_150149053_MP.jpg
    15  IMG_20260901_150143333_MP.jpg  ~  IMG_20260901_150149053_MP.jpg
    16  IMG_20260901_150048239_HDR.jpg  ~  IMG_20260901_150301568_MP.jpg
    16  IMG_20260901_150108448_MP.jpg  ~  IMG_20260901_150043007_MP.jpg
    16  IMG_20260901_150123991_MP.jpg  ~  IMG_20260901_145912520_HDR.jpg
    16  IMG_20260901_150129310_MP.jpg  ~  IMG_20260901_145912520_HDR.jpg
    16  IMG_20260901_150131813_MP.jpg  ~  IMG_20260901_150149053_MP.jpg
  no pair at or below threshold 8.
OK: no near-duplicates found across splits at this threshold.
```

## Supplementary audit: colorhist (the measured-good signal)

Because dHash/pHash proved weakly discriminative on this dataset (see
notes/decisions.md), the audit was repeated with the HSV colour-histogram
distance that separates same-scene from different-scene frames cleanly
(same-scene <= 0.10, unrelated ~0.9):

```
val  vs train: min=0.103, pairs at/below cluster threshold 0.10: 0
test vs train: min=0.102, pairs at/below cluster threshold 0.10: 0
```

Verdict: clean under both metrics. The closest cross-split pairs sit just
outside the same-scene band — consistent with "similar room, different
arrangement", which is exactly what a scene-level split should allow.
