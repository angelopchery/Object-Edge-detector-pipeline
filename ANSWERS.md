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

## Part C

### C1

_TODO_

### C2

_TODO_

### C3

_TODO_

## Part D

### D1

_TODO_

### D2

_TODO_

### D3

_TODO_

### D4

_TODO_

### D5

_TODO_
