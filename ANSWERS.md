# Written Answers

## Part B — Diagnosing three broken snippets

---

### B1 — Snippet 1: Preprocessing and coordinate mapping

#### B1.1 Defects

**Defect 1 — the letterbox padding is never subtracted in postprocess (primary).**

`preprocess` centres the resized image on a 640×640 grey canvas, computing `dh, dw` as the top and left pad. Those offsets are then thrown away: only `r` is returned. `postprocess` divides by `r` but never subtracts the pad first.

The model predicts coordinates in *canvas space*. Mapping back to original-image space requires

```
x_orig = (x_canvas - dw) / r
y_orig = (y_canvas - dh) / r
```

The code computes `x_canvas / r`, so every box is translated by `+dw/r` in x and `+dh/r` in y. Because `dw` and `dh` are fixed for a given aspect ratio, **the error is a constant translation, identical for every box in the frame** — hence systematic, not random. The field engineer's "always a bit off" is exactly the signature of a missing translation term rather than a noisy regressor.

Magnitude, concretely: a 1920×1080 frame gives `r = 640/1920 = 0.333`, `nh = 360`, `dh = (640-360)//2 = 140`. Every box is shifted down by `140 / 0.333 = 420 px` in the original frame. That is not subtle — which tells us something about how this shipped (see Defect 2 and the "why it hides" section).

**Defect 2 — the nominal scale `r` is not the scale actually applied.**

`nh, nw = int(h * r), int(w * r)` truncates. The resize therefore applies a real scale of `nw / w` horizontally and `nh / h` vertically, both of which are ≤ `r` and generally unequal to each other. `postprocess` inverts using `r`, not the applied scale.

This is a *multiplicative* error, so unlike Defect 1 it **grows linearly with distance from the coordinate origin** — the top-left corner. That is the mechanism behind "worse at the edges of the frame." A box whose centre sits near the origin is barely affected; one at the far side of the frame accumulates the full drift.

Worst-case relative error is roughly `1/nh`, so about 0.28% at `nh = 360`. On a 1920 px axis that is ~5 px of drift at the far edge and ~0 px at the near edge. Small on its own, but it is the component that varies with position, and it is what makes the error *worse at the edges* rather than uniformly bad.

**Defect 3 — clipping distorts boxes near the boundary rather than translating them.**

Once Defect 1 has pushed boxes toward one side, the `.clip()` calls clamp the overhanging edge to the image bound while leaving the opposite edge shifted. A box that should be translated is instead *squashed*: its width or height changes. So near the frame edges the failure changes character — from "offset" to "offset and wrong size" — which reads to an operator as the error getting worse there.

**Defect 4 — `postprocess` mutates its input array in place.**

`boxes[:, [0, 2]] /= r` is an in-place divide on the caller's array. Any caller that retains a reference to the original detections sees them silently rescaled, and calling `postprocess` twice on the same array double-scales it. It should copy first.

**Defect 5 — integer dtype hazard.**

If `boxes` arrives as an integer array (common when detections have been rounded for drawing), `/=` either raises a casting error or floor-divides depending on NumPy version and dtype. There is no explicit cast to float.

**Defect 6 — non-contiguous array returned.**

`canvas[:, :, ::-1].transpose(2, 0, 1)` produces a view with negative and permuted strides. `.astype(np.float32)` does materialise a new array, so this happens to be safe here, but the pattern is fragile: reorder those operations and some runtimes (TensorRT, certain ORT execution providers) will read the buffer in the wrong order and silently produce garbage. An explicit `np.ascontiguousarray` documents the requirement.

**Defect 7 — the BGR→RGB conversion is an unverified assumption.**

`[:, :, ::-1]` assumes the model was trained on RGB. If it was trained on BGR the channels are swapped, which typically shows as a modest, hard-to-attribute accuracy loss rather than an obvious failure. Nothing in the code asserts or documents which convention the weights expect.

**Defect 8 — asymmetric padding, unaccounted.**

`(size - nh) // 2` floors, so when `size - nh` is odd the top pad is one pixel smaller than the bottom. Sub-pixel, but it is one more term the inverse mapping does not know about.

#### B1.2 Why it survives casual testing

- **Square and near-square test images make both defects vanish exactly.** If `h == w` then `r = size/h`, `int(h*r) = size` with no truncation, and `dh = dw = 0`. Defect 1's translation is zero and Defect 2's scale drift is zero. Anyone who validated on centre-cropped square samples — or on a dataset of square thumbnails — sees a perfectly correct pipeline. This is precisely why the brief says the bug would be "almost invisible on a square image."
- **mAP is nearly blind to a uniform translation when boxes are large.** A shift of *d* pixels on a box of side *s* costs roughly `1 - d/s` in IoU along that axis. For a big, centred object a moderate shift still clears the 0.5 IoU threshold, so mAP@0.5 barely moves and the regression is invisible on the metric that gets reported.
- **Detection counts, classes and confidences are all correct.** Only the geometry is wrong. Any test asserting "the model found 2 chargers with confidence > 0.8" passes.
- **Demo images are the worst possible test.** They are usually one large, centred object in a well-lit frame — the exact condition under which both the translation and the scale drift are least visible.
- **Defect 2 alone is sub-pixel-to-few-pixel** and would never be noticed without Defect 1 dragging attention to the geometry in the first place.

#### B1.3 Corrected code

```python
import cv2
import numpy as np


def preprocess(img, size=640):
    """Letterbox to (size, size).

    Returns the blob plus the *actual applied* scales and pad offsets, so the
    inverse mapping in postprocess is exact rather than nominal.
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized

    # Note: RGB assumed. Assert this against the training config, do not infer it.
    blob = np.ascontiguousarray(
        canvas[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32
    ) / 255.0

    # The scales actually realised by cv2.resize, not the nominal r.
    scale = (nw / w, nh / h)
    pad = (left, top)
    return blob[None], scale, pad


def postprocess(boxes, scale, pad, orig_shape):
    """Map xyxy boxes from canvas space back to original-image space."""
    boxes = np.asarray(boxes, dtype=np.float32).copy()   # never mutate the caller's array
    sx, sy = scale
    left, top = pad

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / sx
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / sy

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_shape[0])
    return boxes
```

#### B1.4 The test that would have caught it

A **round-trip identity test with no model involved**, parameterised over aspect ratios. This is the single highest-value test in a detection codebase and it takes ten minutes to write.

```python
import numpy as np
import pytest

@pytest.mark.parametrize("h,w", [
    (640, 640),    # square — the case that hides the bug
    (1080, 1920),  # landscape
    (1920, 1080),  # portrait
    (480, 1280),   # extreme wide
    (721, 1279),   # odd dimensions, forces asymmetric pad
])
def test_letterbox_roundtrip(h, w):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _, scale, pad = preprocess(img, size=640)
    sx, sy = scale
    left, top = pad

    # Ground-truth boxes deliberately placed at corners and edges,
    # where the scale-drift defect is largest.
    gt = np.array([
        [0, 0, w - 1, h - 1],                     # full frame
        [0, 0, 20, 20],                           # top-left corner
        [w - 21, h - 21, w - 1, h - 1],           # bottom-right corner
        [w // 2 - 10, h // 2 - 10, w // 2 + 10, h // 2 + 10],  # centre
    ], dtype=np.float32)

    # Forward-map into canvas space exactly as the resize does.
    canvas_boxes = gt.copy()
    canvas_boxes[:, [0, 2]] = gt[:, [0, 2]] * sx + left
    canvas_boxes[:, [1, 3]] = gt[:, [1, 3]] * sy + top

    recovered = postprocess(canvas_boxes, scale, pad, (h, w))
    assert np.max(np.abs(recovered - gt)) < 1.0, "coordinate round-trip drifted > 1 px"
```

Two properties make this test catch what a demo image would not: it includes **non-square** shapes, and it places boxes **at the corners**, where the multiplicative drift is maximal. On the original code it fails on every non-square case by hundreds of pixels.

Complementary checks worth having in CI:

- Assert `preprocess` returns everything needed to invert itself — a signature returning only `r` cannot be correct for a padded resize, and that is reviewable without running anything.
- A visual regression test: render predictions on three fixed frames of differing aspect ratio and diff against committed reference images.
- Log the aspect-ratio distribution of production frames. If the pipeline is only ever validated on 1:1 and deployed on 16:9, that mismatch should be visible in a dashboard, not discovered by a field engineer.

---

### B2 — Snippet 2: Dataset preparation

#### B2.1 Defects

**Defect 1 — augmentation happens before the split, so the validation set leaks (most damaging to the reported metric).**

Each source image produces three entries — original, horizontal flip, brightened — which are appended to one flat list, shuffled, then cut 80/20. Variants of the same photograph land on both sides of the split.

The probability that a given source image contributes at least one variant to train **and** at least one to val is high: with three variants and an 80/20 split, `P(all three land in train) = 0.8³ ≈ 0.51`, so roughly **49% of source images have a near-duplicate spanning the split**. The validation set is measuring memorisation, not generalisation. This is why the reported mAP is healthy and collapses on real data.

**Defect 2 — the horizontal flip does not transform the labels (silently corrupts a third of the training data).**

```python
augmented.append((cv2.flip(img, 1), labels))
```

`cv2.flip(img, 1)` mirrors the pixels left-to-right. `labels` is passed through unchanged. In normalised YOLO xywh, the correct transform is

```
x_center → 1.0 - x_center
```

(and in absolute xyxy, `x1 → W - x2`, `x2 → W - x1`).

Every flipped sample therefore carries boxes at the mirrored-wrong location. Exactly one third of the dataset is mislabelled, and there is no error, no warning, and no visible artefact in the image itself. This is the defect the brief refers to as silently corrupting a third of the training labels.

Note the failure is worst for off-centre objects and *disappears* for objects at the horizontal midpoint — so spot-checking a centred sample confirms nothing.

**Defect 3 — augmentation is baked in statically rather than applied online.**

Every epoch sees the identical flip and the identical `1.3` brightness factor. Static augmentation gives a fixed 3× dataset instead of a fresh stochastic view per epoch, so it provides much weaker regularisation than the same transforms applied in the dataloader — while tripling both disk and RAM cost. The whole decoded dataset is also held in memory as NumPy arrays, which does not survive a real dataset.

**Defect 4 — brightness scaling risks overflow.**

`adjust_brightness(img, 1.3)` on a `uint8` array wraps around rather than saturating unless the implementation explicitly clips. A pixel at 200 becomes 260, which wraps to 4 — bright regions turn black. Whether this bites depends on the unshown implementation, but nothing here guards against it. `cv2.convertScaleAbs` saturates correctly; naive NumPy multiplication does not.

**Defect 5 — augmented images are present in the validation set.**

Even with the leak fixed, validation should be measured on unmodified images. Reporting metrics over synthetically brightened and flipped frames measures performance on a distribution that does not exist in production.

**Defect 6 — the shuffle is unseeded.**

`random.shuffle` without a seed makes the split non-reproducible. Two runs of the same script produce different train/val sets and therefore different metrics, making regressions impossible to attribute.

**Defect 7 — `cv2.imread` failures pass silently.**

A corrupt or unreadable file returns `None`, which is appended without complaint and fails much later with an opaque error inside the training loop.

**Defect 8 — the glob only matches `*.jpg`.**

`.jpeg`, `.png` and `.JPG` files are silently excluded. The dataset is quietly smaller than the annotator believes, and the omission is invisible because the count printed at the end is of *augmented* items, not source images.

**Defect 9 — no class-balance check across the split.**

A random 80/20 cut on a small dataset can leave one class badly under-represented, or absent, in validation. Nothing here checks.

#### B2.2 Why it survives casual testing

- **The printed output is exactly what you expect.** `train: 360 val: 90` looks like a healthy split. Nothing in the console reveals that those 90 are near-copies of the 360.
- **The validation metric *rises*, and rising metrics do not get investigated.** The leak inflates mAP; the label corruption depresses it. The net reported number is still good, so there is no trigger to look closer.
- **The two defects interact to hide each other, which is the subtle part.** The model trains on flipped images with un-flipped labels, so it learns the corrupted mapping. Validation *also* contains flipped images with the same corrupted labels. The model's wrong predictions therefore *agree with the wrong ground truth*, and score as correct. The metric is self-consistently wrong. Only real, unflipped, unseen data breaks the agreement — which is precisely when the collapse is reported.
- **Nothing raises.** Every defect is semantic. There is no exception, no shape mismatch, no NaN.
- **Visual inspection of a flipped sample looks fine unless boxes are rendered.** And if you happen to render a near-centred object, the mirrored box still overlaps it.

#### B2.3 Which single defect does the most damage to the reported metric, and why

**The leak (Defect 1).**

The distinction matters and is worth being precise about. The flip-label bug (Defect 2) is the more serious *engineering* error — it corrupts a third of the training signal and genuinely degrades the model. But its effect on the *reported number* is to push it **down**. It cannot explain a healthy validation mAP.

Only the leak inflates the metric. It severs the connection between the reported score and any statement about unseen data: the model has effectively been evaluated on its own training set. A model can be arbitrarily bad and still score well, and that is what makes it the defect responsible for the collapse-on-contact-with-reality symptom.

The second-order interaction described above amplifies this — the corrupted labels are *consistent* across the leaked split, so the model is rewarded for reproducing them. Fixing only the flip while leaving the leak in place would still produce an untrustworthy number.

#### B2.4 Corrected code

```python
import glob
import os
import random

import cv2
import numpy as np

SEED = 42
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def flip_labels_horizontal(labels):
    """Mirror normalised YOLO xywh labels about the vertical axis."""
    flipped = []
    for cls, xc, yc, bw, bh in labels:
        flipped.append((cls, 1.0 - xc, yc, bw, bh))
    return flipped


def build_splits(image_dir, val_frac=0.2, seed=SEED):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError(f"no images found under {image_dir}")

    # --- SPLIT FIRST, on source identity, before any augmentation ---
    rng = random.Random(seed)
    rng.shuffle(paths)
    n_val = int(round(val_frac * len(paths)))
    val_paths, train_paths = paths[:n_val], paths[n_val:]

    assert not (set(train_paths) & set(val_paths)), "source image present in both splits"

    def load(path):
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"failed to decode {path}")
        return img, load_labels(path)

    # Validation: originals only, never augmented.
    val = [load(p) for p in val_paths]

    # Train: augment, transforming labels alongside the pixels.
    train = []
    for p in train_paths:
        img, labels = load(p)
        train.append((img, labels))
        train.append((cv2.flip(img, 1), flip_labels_horizontal(labels)))
        train.append((cv2.convertScaleAbs(img, alpha=1.3, beta=0), labels))  # saturating

    print(f"source images: {len(paths)}  train src: {len(train_paths)}  val src: {len(val_paths)}")
    print(f"train samples after augmentation: {len(train)}  val samples: {len(val)}")
    return train, val
```

Two notes on this correction:

- `cv2.convertScaleAbs` saturates at 255 instead of wrapping, fixing Defect 4. Brightness is left un-transformed in the labels because it is photometric, not geometric — worth stating explicitly so the asymmetry with the flip is deliberate rather than accidental.
- In production I would delete the static augmentation entirely and move flip/brightness into the dataloader, so each epoch draws a fresh random transform. The version above is kept close to the original to make the fix legible.

**A stronger split than the one shown.** For a dataset captured as multiple near-duplicate frames per physical arrangement — which is how small self-captured datasets are built, including my own in Part A — splitting on *file* identity is still not enough, because scene-mates are near-duplicates of each other. The split must be on **capture scene**, so no frame from a given arrangement appears on both sides. That is what `scripts/split_dataset.py` in this repository does, and the reason is exactly the failure mode this snippet demonstrates in a milder form.

#### B2.5 The tests that would have caught it

**For the leak — an assertion, not a test.** Derive a source identity for every sample and assert the train and val identity sets are disjoint. This belongs in the split function itself so it cannot be skipped:

```python
def test_no_source_leak_across_split():
    train, val = build_splits("dataset/images")
    train_ids = {sample_source_id(s) for s in train}
    val_ids = {sample_source_id(s) for s in val}
    assert train_ids.isdisjoint(val_ids)
```

A blunter version that catches leakage even when identity tracking is wrong: hash every image array in both splits and assert no hash collides. And for near-duplicates rather than exact ones, compute a perceptual hash (or embedding cosine similarity) between every val image and every train image, and flag any pair above a similarity threshold. On the original code this fires on roughly half the dataset.

**For the flip — a geometric round-trip test.**

```python
def test_horizontal_flip_transforms_labels():
    img = cv2.imread("tests/fixtures/off_centre_object.jpg")
    labels = load_labels("tests/fixtures/off_centre_object.jpg")

    f_img = cv2.flip(img, 1)
    f_labels = flip_labels_horizontal(labels)

    # Flipping twice must return to the original.
    assert np.allclose(
        [l[1] for l in flip_labels_horizontal(f_labels)],
        [l[1] for l in labels],
    )

    # And the box must still land on the object: crop both and compare
    # mean pixel intensity inside the box against the background.
    assert box_contains_object(f_img, f_labels[0])
```

The fixture must be **off-centre** — a centred object passes even with the bug.

**The cheapest and most effective check of all:** a script that renders every augmented sample with its boxes drawn and writes a contact sheet to disk. Five minutes of scrolling through it makes a mirrored-box bug unmissable. Any augmentation pipeline should ship with one, and it should be run every time a transform is added.

---

### B3 — Snippet 3: IoU and non-maximum suppression

#### B3.1 Defects

**Defect 1 — the intersection is never clamped to zero (root cause of the disappearing boxes).**

```python
inter = (x2 - x1) * (y2 - y1)
```

For boxes that do not overlap, `x2 - x1` is negative, and so is `y2 - y1`. Their product is **positive**. The function reports a large intersection for two boxes that share no area at all.

Crucially, the spurious intersection *grows with separation*: the further apart the boxes, the more negative each term, and the larger the fake overlap. The correct form clamps each side independently before multiplying.

**Defect 2 — the area formula treats xyxy coordinates as xywh.**

```python
area1 = box[2] * box[3]
area2 = boxes[:, 2] * boxes[:, 3]
```

With boxes in xyxy, `box[2]` is `x2` and `box[3]` is `y2`. The product is not the area — it is the product of the bottom-right corner coordinates. Correct is `(x2 - x1) * (y2 - y1)`.

This inflates areas enormously for boxes far from the origin, and it interacts with Defect 1: since both numerator and denominator are wrong, the resulting ratio is not bounded to [0, 1] and has no geometric meaning.

**Defect 3 — NMS is class-agnostic; the `classes` argument is accepted and ignored.**

Suppression runs over all detections regardless of class. Two genuinely distinct objects of different classes whose boxes overlap — a charger brick partly behind an earphone case, a person holding a tool — result in the lower-scoring one being deleted, even though both are correct.

`classes` should be used to run suppression **independently per class**. The standard vectorised idiom is to offset each box by `class_id × (max_coordinate + 1)` so boxes of different classes can never overlap, then run a single global NMS pass — but an explicit per-class loop is clearer and is what I would write here.

**Defect 4 — no confidence threshold before suppression.**

Every proposal enters NMS, including near-zero-scoring ones. This is a performance problem (the loop is O(n²) in kept boxes) and a correctness one: a spurious low-confidence box that survives can suppress a real one, since order of suppression depends only on relative score.

**Defect 5 — division by zero.**

`inter / (area1 + area2 - inter)` has no epsilon. A degenerate box of zero area produces `0/0 → nan`, and `nan < thr` evaluates `False`, so the comparison silently keeps everything. No warning is raised by default in a NumPy build with errstate suppressed.

**Defect 6 — no validation that boxes are well-formed.**

Nothing asserts `x1 < x2` and `y1 < y2`. A box that arrives inverted (from an unclamped regression output, or a coordinate-mapping bug like Snippet 1's) propagates nonsense through the whole function.

**Defect 7 — `keep` order and index semantics are undocumented.**

The returned indices refer to the original arrays, and are ordered by descending score. That is the correct convention, but it is unstated, and a caller who assumes indices into a filtered array gets wrong boxes with no error.

For completeness: the loop mechanics themselves — `order = order[1:][ious < thr]` — are the standard correct formulation. That is not a defect.

#### B3.2 The mechanism that makes a valid detection disappear

Defects 1 and 2 combine to produce a spurious IoU above threshold for two boxes that are **small, well-separated, and offset diagonally**.

A worked example, which is reproducible in a REPL:

```
Box A (kept, higher score):  (  0,   0, 100, 100)
Box B (valid, lower score):  (500, 500, 600, 600)

x1 = max(0, 500) = 500      x2 = min(100, 600) = 100
y1 = max(0, 500) = 500      y2 = min(100, 600) = 100

inter = (100 - 500) * (100 - 500) = (-400) * (-400) = 160000   ← positive, and large
area1 = 100 * 100  = 10000     (should be 10000 — coincidentally right at the origin)
area2 = 600 * 600  = 360000    (should be 10000 — inflated 36×)

IoU = 160000 / (10000 + 360000 - 160000) = 160000 / 210000 = 0.76
```

0.76 exceeds the 0.5 threshold, so **box B is suppressed**. Two objects 500 pixels apart, sharing not a single pixel, and the second one vanishes. The operator sees a clearly visible object with no box on it.

#### B3.3 Why it happens more often in sparse frames than crowded ones

This is the counter-intuitive part, and it follows directly from the sign arithmetic.

The spurious intersection requires **both** `(x2 - x1)` and `(y2 - y1)` to be negative, so their product is positive. That is the geometry of two boxes separated along *both* axes simultaneously — diagonally offset, with no overlap in either projection.

- In a **crowded frame**, candidate boxes cluster tightly. Neighbouring boxes typically overlap in at least one axis: they sit side by side (overlapping in y) or stacked (overlapping in x). One difference is positive, the other negative, so `inter` is **negative**, the IoU is negative, `ious < thr` is `True`, and the box is **kept**. The bug does not suppress here. If anything it *under*-suppresses, leaving duplicate boxes on the same object.
- In a **sparse frame**, the few objects present are far apart and generically offset in both x and y. Both differences go negative, `inter` goes large and positive, and the spurious IoU clears the threshold. The bug **over**-suppresses.

The magnitude reinforces the pattern: the fake intersection scales with the product of the separations, so the further apart the objects, the more certain the suppression. Sparse frames are exactly the frames with large separations.

This also explains why "detection counts look reasonable in aggregate." The two failure modes push in opposite directions — spurious duplicates retained in dense frames, valid boxes deleted in sparse ones — so a count averaged over a mixed dataset comes out approximately right while individual frames are wrong in both directions.

#### B3.4 What the `classes` argument should have been doing

It should scope suppression to within a class. NMS exists to collapse multiple proposals for *the same physical object*; two detections of different classes are by construction not the same object, and one must never remove the other.

The bug is invisible in aggregate metrics because per-class AP is computed independently, and it only bites when two different-class objects genuinely overlap in the image — which, in a dataset like mine where a charger brick and an earphone case are frequently adjacent or stacked, is common rather than rare.

#### B3.5 Corrected code

```python
import numpy as np


def iou(box, boxes, eps=1e-9):
    """IoU of one xyxy box against an array of xyxy boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # Clamp each side independently. Multiplying two negative
    # differences would otherwise report a large fake overlap.
    inter_w = np.clip(x2 - x1, 0.0, None)
    inter_h = np.clip(y2 - y1, 0.0, None)
    inter = inter_w * inter_h

    # xyxy areas: (x2 - x1) * (y2 - y1), not x2 * y2.
    area1 = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
    area2 = (np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
             * np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None))

    return inter / (area1 + area2 - inter + eps)


def nms(boxes, scores, classes, thr=0.5, score_thr=0.25):
    """Per-class non-maximum suppression.

    Returns indices into the original arrays, highest score first.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    classes = np.asarray(classes)

    if not (boxes[:, 2] >= boxes[:, 0]).all() or not (boxes[:, 3] >= boxes[:, 1]).all():
        raise ValueError("malformed box: requires x1 <= x2 and y1 <= y2")

    keep = []
    for c in np.unique(classes):
        # Suppression is scoped to one class: a brick must never
        # suppress a case, they are not the same object.
        idx = np.where((classes == c) & (scores >= score_thr))[0]
        order = idx[np.argsort(scores[idx])[::-1]]

        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            ious = iou(boxes[i], boxes[order[1:]])
            order = order[1:][ious < thr]

    return sorted(keep, key=lambda i: -scores[i])
```

#### B3.6 The tests that would have caught it

**Unit-test `iou` against hand-computed values.** This is thirty seconds of work and catches both Defects 1 and 2 immediately:

```python
def test_iou_known_values():
    box = np.array([0, 0, 10, 10], dtype=np.float32)

    # Identical box → 1.0
    assert np.isclose(iou(box, np.array([[0, 0, 10, 10]], dtype=np.float32))[0], 1.0)

    # Disjoint, diagonally separated → 0.0  (fails on the original: returns 0.76)
    assert np.isclose(iou(box, np.array([[50, 50, 60, 60]], dtype=np.float32))[0], 0.0)

    # Disjoint, separated in x only → 0.0
    assert np.isclose(iou(box, np.array([[20, 0, 30, 10]], dtype=np.float32))[0], 0.0)

    # Half overlap: inter 50, union 150 → 1/3
    assert np.isclose(iou(box, np.array([[5, 0, 15, 10]], dtype=np.float32))[0], 1 / 3)

    # Translation invariance — catches the xyxy-as-xywh area bug specifically.
    far = np.array([1000, 1000, 1010, 1010], dtype=np.float32)
    assert np.isclose(
        iou(box, np.array([[5, 0, 15, 10]], dtype=np.float32))[0],
        iou(far, np.array([[1005, 1000, 1015, 1010]], dtype=np.float32))[0],
    )
```

The translation-invariance case is the one I would insist on in review: IoU is a purely relative quantity, so an identical configuration translated by 1000 px must give an identical result. The original code fails it badly, and the failure points straight at the area formula.

**A property test over random boxes**, asserting the invariant that no correct implementation can violate:

```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers(0, 2000), min_size=8, max_size=8))
def test_iou_always_in_unit_interval(vals):
    a = np.array([min(vals[0], vals[2]), min(vals[1], vals[3]),
                  max(vals[0], vals[2]), max(vals[1], vals[3])], dtype=np.float32)
    b = a.copy().reshape(1, 4)
    b[0] = [min(vals[4], vals[6]), min(vals[5], vals[7]),
            max(vals[4], vals[6]), max(vals[5], vals[7])]
    v = iou(a, b)[0]
    assert 0.0 <= v <= 1.0 + 1e-6
```

Random sampling finds the disjoint-diagonal case within a handful of draws.

**For the class-agnostic bug**, a direct behavioural test:

```python
def test_nms_does_not_suppress_across_classes():
    boxes = np.array([[0, 0, 100, 100], [10, 10, 110, 110]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 1])           # different classes, heavily overlapping
    assert len(nms(boxes, scores, classes)) == 2

    classes = np.array([0, 0])           # same class → one should be suppressed
    assert len(nms(boxes, scores, classes)) == 1
```

**And at the system level**, the check that would have surfaced this without anyone suspecting NMS at all: log the count of detections **before and after** suppression, per frame, and alert on frames where the ratio is anomalous. In sparse frames the original code deletes boxes it should not, so the suppression ratio there is an outlier against an otherwise stable distribution. That signal costs nothing to emit and turns "operators occasionally report a missing box" from an anecdote into a queryable event.

## Part C — Production failure diagnoses

### C1 — INT8 quantisation collapse: 0.91 → 0.58 mAP

The size of the drop matters diagnostically. Quantisation noise on a healthy
model costs points, not thirds. A 0.33 mAP collapse means something is
*structurally* wrong — a whole code path, layer group, or data distribution is
broken, not just rounding error. That rules out "INT8 is just like that" as an
explanation before we start.

**Order of investigation, cheapest and most-often-guilty first.**

**Check 0 — is it even the quantisation?** Run the FP32 model through the
*deployment* runtime (TensorRT FP32 engine, same preprocessing code) on the
same eval set. If FP32-through-TensorRT also scores ~0.58, quantisation is
innocent and this is a preprocessing/pipeline divergence between the training
framework and the deployment stack — wrong normalisation, BGR/RGB swap,
different letterbox pad or resize interpolation. This is the most common
"quantisation bug" in practice and it is not a quantisation bug at all.
Distinguishing test: FP32 deployment-path evaluation. Costs one engine build.
(This repo's `verify_parity.py` exists precisely to catch this class before it
ships: same tensor through both runtimes, max abs diff on raw outputs, then
post-NMS box agreement in pixels.)

**Cause 1 — unrepresentative calibration set.** Static INT8 chooses activation
ranges from calibration images. If they were dev-box screenshots, a handful of
easy frames, or images from a different site/camera than deployment,
activation clipping destroys exactly the feature ranges real data needs.
Distinguishing test: recalibrate with 100+ images sampled from the actual
deployment distribution and re-evaluate — a large recovery convicts
calibration. Corroborating evidence: per-image score degradation concentrates
in conditions (lighting, scale) absent from the calibration set.
(In this repository calibration draws only from the training split and
`quantize.py` refuses val/test paths — using eval images for calibration is a
leak that flatters the INT8 number.)

**Cause 2 — per-tensor weight quantisation on a depthwise-separable
backbone.** Depthwise conv layers have wildly different per-channel weight
ranges; one 8-bit scale per tensor crushes the small-range channels to zero.
Distinguishing test: rebuild with per-channel weight quantisation (both ORT
and TensorRT support it) and compare — recovery convicts the scheme, not the
data. Corroborating evidence: dump per-layer weight ranges; a layer whose
max/min channel-range ratio is in the hundreds is the smoking gun.

**Cause 3 — the detection head does not survive INT8.** Box-regression
outputs (especially DFL-style distributions) and objectness logits have value
ranges and sensitivities that 8 bits represent poorly. Distinguishing test:
exclude the head (final conv layers) from quantisation, keep the backbone
INT8, re-evaluate. Recovery with negligible latency cost convicts the head.
This is also the fix with the best accuracy-per-millisecond trade-off.

**Cause 4 — silent layer fallback / botched fusion.** TensorRT falls back
unsupported ops to FP32 and fuses layers around Q/DQ nodes; a badly placed
Q/DQ pair can leave a subgraph quantised in a numerically hostile spot.
Distinguishing test: inspect the engine build log / layer-precision report —
which layers actually run INT8 — and compare intermediate activations
layer-by-layer between FP32 and INT8 on one image; the first layer where
cosine similarity craters localises the defect.

**Validation before it reaches the client:** evaluate FP32 and INT8 with the
*same* evaluation code on the same frozen split (in this repo:
`evaluate_onnx.py` for both, cross-checked against Ultralytics in Phase 6),
report the drop as a number alongside latency and size, and gate the release
on a maximum acceptable drop agreed in advance.

*Grounding from this repository:* the measurement on my own model —
static INT8 (QDQ, per-channel, MinMax, train-split calibration) — is recorded
in the README trade-off table; per-channel weight quantisation was the default
precisely because of Cause 2, and the measured drop is quoted there rather
than re-typed here so the two can never disagree.

### C2 — One camera of twelve, consistent directional offset, worse at edges

**What the pattern alone localises.** Twelve cameras share one model. A model
defect would degrade all twelve; eleven are fine, so the model is innocent. A
loose mount or random noise would not produce a *consistent direction*; a
constant directional shift is a translation error. "Worse at the frame edges"
is the signature of a multiplicative (scale) error, which is ~zero at the
coordinate origin and grows linearly toward the far corner. Both signatures at
once — uniform offset plus edge-growing error — is precisely the letterbox
inverse-mapping defect: pad not subtracted (translation) and nominal-vs-actual
scale mismatch (edge-growing drift). This is the same defect class as
Snippet 1 (Part B1), where the arithmetic is worked in detail: the per-stream
coordinate transform for that camera is wrong, almost certainly because that
camera's resolution or aspect ratio differs from the other eleven and its
letterbox parameters are stale, hardcoded, or computed from the wrong
dimensions.

**Confirming without physical access, in order:**
1. Pull the stream metadata: compare that camera's advertised resolution and
   aspect ratio against the other eleven. A lone 4:3 camera in a 16:9 fleet,
   or a substream at a different resolution, closes the case at step one.
2. Grab one frame plus one detection from the live pipeline, and run the same
   frame through the reference implementation offline. The difference between
   the two box sets *is* the transform error; a constant delta measures the
   pad term, and a delta growing with distance from the origin measures the
   scale term.
3. Round-trip test with no model at all (the B1.4 test): forward-map synthetic
   corner boxes through that stream's preprocess parameters and invert them.
   Drift over 1 px reproduces the bug deterministically.
4. Project a known fixed point (doorframe corner, floor marking visible in the
   frame) through the pipeline and compare its reported position across
   cameras.

**Fix and validation.** Compute letterbox scale and pad per-stream from the
actual decoded frame dimensions (never from config constants), return the
*applied* scale (post-rounding) rather than the nominal ratio, and add the
parameterised round-trip test over every resolution in the fleet to CI so the
next odd camera cannot ship. Validate by re-projecting the fixed reference
point on all twelve streams and confirming sub-pixel agreement.

### C3 — 97% → 84% over three months, "nothing changed"

"Nothing changed" means "nothing anyone chose to change" — three months of
gradual decay is the signature of the environment changing under a frozen
model. Three causes cover most real cases; each has cheap confirming evidence
in the logs and images the system already produces.

**Cause 1 — illumination drift.** Seasonal daylight shift, aging or failed
luminaires, a relamped hall. Confirming evidence: per-camera mean frame
brightness and histogram, trended over the three months — a monotonic drift or
a step at a specific date convicts lighting. (This dataset's own variation
audit measured brightness mean 126, std 21.8, range 83–170 — a model trained
on one lighting regime, as mine was, is maximally exposed to exactly this
failure.)

**Cause 2 — camera physical degradation.** Dust or film on the lens, focus
creep, mount drift changing the viewing angle. Confirming evidence: image
sharpness (variance of Laplacian) trended per camera; projection of a fixed
scene landmark over time; degradation isolated to specific cameras rather
than uniform across the fleet.

**Cause 3 — input distribution change.** New product variant, changed
packaging, new pallet/backdrop, workers placing items differently. Confirming
evidence: rising fraction of low-confidence detections, per-class recall
falling asymmetrically, and human review of a sample of recent misses — if the
missed objects are visibly "new" (my own dataset contains a black charger and
a white charger; a model trained only on the white one would exhibit exactly
this on the black), the case is closed.

**The monitoring signal that would have fired inside two weeks.** Accuracy
cannot be monitored directly without labels, but its proxies can:

- Log per-frame **mean detection confidence** and **detections-per-frame**,
  aggregated per camera per day.
- Commission baseline: first 30 days of deployment give a mean and sigma per
  camera (e.g. confidence 0.82, sigma 0.015 at daily aggregation).
- **Alert rule: 7-day rolling mean confidence more than 2 sigma below the
  commissioning baseline, per camera** — with a 13-point accuracy decay over
  ~12 weeks (≈1.1 points/week) and confidence tracking accuracy, the rolling
  mean crosses a 2-sigma band (~0.03) inside two weeks of the decay starting,
  while day-to-day noise stays inside it.
- Secondary rule for the physical causes: per-camera mean brightness or
  sharpness ±20% from commissioning baseline for 3 consecutive days.

Both rules are one SQL query over logs the system should already be writing;
neither needs a single labelled frame.

## Part D — Edge deployment: 8 cameras, 200 ms, air-gapped

### D1 — Throughput vs latency: the two budgets

Aggregate throughput: **8 cameras × 15 fps = 120 frames/second, sustained.**

These are two different constraints and they fail differently:

- **Throughput budget:** the box must *complete* at least 120 inferences per
  second indefinitely. If one inference occupies the accelerator for t ms,
  serial execution needs t ≤ 1000/120 = **8.3 ms**; with batching or multiple
  execution streams, the requirement is that *amortised* per-frame occupancy
  stays under 8.3 ms.
- **Latency budget:** each individual frame must go decode → preprocess →
  inference → postprocess → action in ≤ **200 ms** end-to-end. Inference is
  only one term; at 15 fps a frame can also wait up to 66.7 ms for its slot in
  a batch, and decode/NMS/serialisation all bill against the same 200 ms.

Batching illustrates why conflating them is the standard error: batch-8
inference at, say, 40 ms per batch gives amortised 5 ms/frame — throughput
passes comfortably — but a frame arriving just after a batch closes waits up
to 66.7 ms (one frame interval) before its batch even starts, then 40 ms of
compute: batching *helps throughput and hurts per-frame latency*. With a
200 ms budget there is room for moderate batching, but the sum
(wait + decode + preprocess + inference + postprocess + action) must be
measured as a distribution — p95/p99, not mean — because the budget is
per-frame, not on average.

### D2 — Model, precision, and the first measurement

Choice: a **nano/small single-stage detector (YOLO11n/s class) at INT8** on
the edge accelerator, with the detection head kept at higher precision if the
INT8 accuracy drop measured on-device exceeds the agreed budget (see C1
Cause 3), and FP16 as the documented fallback.

Why the combination fits: a fixed camera, a small closed set of rigid object
classes, and large-ish objects is exactly the regime where detector capacity
stops mattering and the nano tier is sufficient — my own 5.4 MB YOLO11n on
two classes supports this. INT8 quarters memory bandwidth and roughly doubles
throughput on edge accelerators versus FP16, which is what makes 120 fps
sustained plausible on a single low-power device.

The single first measurement: **sustained end-to-end p95 latency and
throughput of the actual INT8 engine on the actual target device, under
thermal steady-state (30+ minutes), at the batch size the pipeline will
really use** — not a datasheet TOPS figure, not a desktop benchmark. My
benchmark methodology in this repo (20 warmup + 200 timed iterations, p95 and
std reported, execution provider and thread count logged) is the template; on
a laptop I additionally noted that thermal throttling makes short benchmarks
unreproducible, and an edge box in a warm factory corner is worse. I would
benchmark before committing to any specific device count rather than assert
one here.

### D3 — Air-gapped retraining loop

1. **Flag at the line.** The operator UI has one control: "this was wrong"
   (missed object / wrong box / false alarm). The system stores the frame,
   the model's raw predictions, model version, camera ID and timestamp in a
   local ring buffer. Low-confidence detections are auto-queued for review as
   well — operators only catch what they notice.
2. **Cross the gap physically.** On a schedule (weekly, or when the buffer
   fills), flagged bundles are exported to removable media — hashed and
   signed so the lab can verify integrity — and walked across the gap.
   Nothing else crosses.
3. **In the lab:** review every flagged frame against the annotation guide
   (the same discipline as this repo: written rules, rendered-box review,
   verify_labels-style gates), add to the training corpus with scene/site
   metadata, retrain, and evaluate on (a) a frozen held-out test set that
   never trains and (b) a regression suite built from previously-fixed
   failures, so old fixes cannot silently regress.
4. **Return crossing:** the candidate model ships back as a signed package
   (weights + config + expected-output vectors for N canonical frames). The
   edge box verifies the signature and runs the canonical frames, refusing a
   package whose outputs do not match — a parity check in the same spirit as
   verify_parity.py.
5. **Staged replacement:** the candidate runs in shadow mode alongside the
   incumbent for a fixed soak period (disagreement rate logged), then swaps
   in only if shadow metrics clear the bar. The incumbent is retained for
   rollback (D4).

### D4 — Rollback

- **Mechanism:** the previous model version stays on disk; the runtime loads
  models via an atomic pointer (symlink/config swap) so rollback is a restart
  of the inference process, not a redeployment — seconds, executable by a
  site technician or automatically.
- **Trigger signal:** the C3 monitors, tightened for the post-deploy window —
  per-camera 24-hour rolling mean detection confidence or detections-per-frame
  outside ±2 sigma of the pre-swap baseline, or the shadow-period disagreement
  rate jumping after the swap. Plus a manual trigger for operators.
- **Detection latency:** with 120 fps there is no shortage of samples — at
  hourly aggregation, a C1-style collapse (tens of points) is detectable
  within 1–2 hours of the swap; a slow C3-style drift within days. The swap
  is scheduled at shift start so the first monitored window has production
  traffic.
- **Drill:** rollback that has never been exercised does not exist; it is
  rehearsed at every model update by design (the swap procedure and the
  rollback procedure are the same code path in opposite directions).

### D5 — What I am least confident about, honestly

**The quantised model's behaviour on the target accelerator's toolchain.**
Everything in this repository quantised with ONNX Runtime on x86; a real
edge deployment (TensorRT on Jetson, or an NPU vendor toolchain) requantises
with a different calibrator, different kernel fusions and different
per-layer precision decisions. My ORT INT8 numbers bound nothing about a
TensorRT INT8 engine — C1 is a whole answer about the ways that specific
translation fails. Second: **sustained thermal behaviour** — my latency
figures came from a laptop that I explicitly could not hold in thermal
steady-state, and I flagged them as such in the README.

What resolves both is the same thing: the target device on a desk, the real
engine built by the real toolchain, the frozen test set evaluated on-device,
and a 30-minute sustained-load benchmark — before any commitment about
device count or frame budget is made to the client. I would rather deliver
that measurement plan than a confident number I do not have.
