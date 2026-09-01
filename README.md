# Object Detection: earphone_case / charger_brick

Two-class detector (YOLO11n) on a self-captured dataset, exported to ONNX,
quantised to INT8, benchmarked — with a visually-derived scene-level split,
a two-metric leakage audit, and a held-out test set evaluated exactly once.

> **Convention:** every number below was produced by a command in this repo,
> run on the hardware listed; the producing command is named beside each
> table. Nothing is estimated. The full decision log is in
> [notes/decisions.md](notes/decisions.md).

## Headline results (and why this ordering)

| Metric | Val | Held-out test (run once) |
|---|---|---|
| **mAP@0.5:0.95 (FP32)** | **0.722** | **0.659** |
| mAP@0.5 (FP32) | 0.956 | 0.849 |
| mAP@0.5:0.95 (INT8) | 0.689 | 0.609 |
| mAP@0.5 (INT8) | 0.933 | 0.793 |

(`scripts/evaluate_onnx.py --split val|test`, identical code for all four.)

**mAP@0.5:0.95 leads because mAP@0.5 is not the discriminator here.** The
median box is ~8% of frame area and images hold 1–2 large objects; IoU 0.5 is
easy to clear, so mAP@0.5 saturates (0.99 on val under Ultralytics' own
validator). The 0.5:0.95 figure and the **val→test gap (−0.063 / −0.107)**
are the honest signals: the 21-image val split was optimistic by roughly that
much.

## Hardware & environment

Quoted from [notes/environment.md](notes/environment.md), captured from the
running venv (Python 3.11.9):

```
torch 2.5.1+cu121 | cuda True | NVIDIA GeForce RTX 3050 Laptop GPU
torchvision 0.20.1+cu121
ultralytics 8.4.137
onnx 1.22.0 | onnxruntime 1.29.0 ['AzureExecutionProvider', 'CPUExecutionProvider']
numpy 2.4.6 | opencv 5.0.0 | pillow 12.3.0
```

RTX 3050 Laptop GPU (4 GB VRAM), Intel i5-12450H, 16 GB RAM, Windows 11.
Two environment defects were caught during setup (a dead pip extra-index in
the machine-wide pip.ini; `pip install -r requirements.txt` silently
replacing CUDA torch with a CPU build) — both diagnosed in
[notes/decisions.md](notes/decisions.md).

## Dataset

- **150 images**, photographed by me in one session (14:50–15:04, phone
  camera, both orientations), resized to 1280 px long edge
  (`scripts/resize_images.py`, EXIF baked in).
- **208 boxes**: 95 `earphone_case` (id 0), 113 `charger_brick` (id 1),
  1–2 boxes per image. **Four physical objects**: two chargers (black,
  white) and two cases — at most one unit per class per frame.
- Labelled in makesense.ai; YOLO txt is authoritative downstream. The VOC
  XML and CSV exports are committed as provenance only (absolute pixel
  coordinates that go stale after resize).
- `scripts/verify_labels.py`: **0 hard errors** — no orphans either way, all
  coordinates in [0,1], no degenerate boxes, class IDs only {0,1}.
- EXIF gate (`scripts/check_exif_orientation.py`): all 150 EXIF-applied
  sizes match the CSV labelling dims; **147/150 images carry an EXIF
  rotation**, so baking orientation into the resize was load-bearing.

### The caught class-ID defect (worth reading)

The scaffolded `data.yaml` assumed `0=charger_brick, 1=earphone_case`. Before
training, the YOLO txt export was cross-referenced against the CSV export by
matching all 208 boxes coordinate-by-coordinate: the measured mapping is
**0=EarphoneCase, 1=ChargingCase — the reverse**. Flipped class IDs train
cleanly and produce plausible metrics; nothing at runtime would have failed.
Caught by cross-referencing export formats, not by testing (commit 7324505),
then confirmed visually against crops in `notes/class_check/`.

### Split: scenes derived visually, gated by a leakage audit

Filenames are camera originals (no scene encoding), and timestamp clustering
cannot separate an 87-image continuous burst, so scene identity was derived
from image content (`scripts/cluster_scenes.py`):

- 64-bit pHash was **rejected by measurement** — temporally adjacent frames
  averaged Hamming 29.4/64 vs 31.5 for frames >60 s apart (no separation).
- HSV colour-histogram correlation separates cleanly (adjacent median
  distance 0.236 vs 0.916) and is the default; threshold 0.10 chosen from
  an auditable scan ([notes/scene_clustering.md](notes/scene_clustering.md))
  → **40 clusters**.
- Split (`scripts/split_dataset.py --test-frac 0.33 --val-frac 0.2 --seed 42`,
  whole clusters only, greedy-balanced): **train 79 / val 21 / test 50**
  images (46/21/28 case, 59/18/36 brick). Manifest committed at
  `data/split_manifest.json`.
- **Leakage audit** (`scripts/check_leakage.py`,
  [notes/leakage_report.md](notes/leakage_report.md)): clean under dHash
  (closest cross-split pair Hamming 10, threshold 8) **and** under the
  colorhist metric (min cross-split distance 0.102 vs same-scene ≤0.10).
  After training tripped the mAP@0.5>0.97 tripwire, a tighter re-audit
  flagged 3 pairs; each was inspected visually — same room, *different
  object and arrangement* — allowed by a scene split, not leakage.

### Annotation

Rules in [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md). Process honesty: all
150 images were hand-labelled in one pass **before** the guide was finalised
— the guide was then written *descriptively* from the labels (each rule
marked `[inferred]` with its evidence) and confirmed by me. No seed model or
pre-labelling was used (`prelabel.py` exists but played no part). This
ordering is a failed process gate, recorded in Known Gaps rather than hidden.

## Training

`python scripts/train.py --data data/data.yaml --model yolo11n.pt --imgsz 640
--batch 8 --epochs 100 --patience 25 --name final` — full resolved config as
actually run: [runs/detect/final/resolved_config.json](runs/detect/final/resolved_config.json)
(committed; key values: imgsz 640, batch 8, AMP on, seed 42, optimizer auto).

- **87 epochs** (early stop, best at 62), **11 m 14 s wall-clock**, peak
  ~1.4 GB VRAM of 4 GB.
- First launch failed and is kept in history (commit 8f5f903): the machine's
  global Ultralytics `datasets_dir` pointed at a different project, so
  relative dataset paths resolved outside this repo. `train.py` now resolves
  the dataset yaml and run dir absolutely.

### Validation (best.pt, Ultralytics validator)

| Class | P | R | mAP@0.5 | mAP@0.5:0.95 |
|-------|---|---|---------|--------------|
| earphone_case | 1.000 | 0.841 | 0.985 | 0.769 |
| charger_brick | 0.888 | 1.000 | 0.990 | 0.742 |
| all | 0.944 | 0.921 | 0.988 | 0.755 |

### Evaluation cross-check (Phase 6)

The custom evaluator was cross-validated against Ultralytics on the identical
val split: 0.956/0.722 vs 0.988/0.755 — a 0.032 gap. The NMS IoU convention
was tested and eliminated (0.45→0.7 moved it 0.001); the residual is
rect-batched val preprocessing vs fixed square-640 letterbox plus AP
interpolation (101-point vs all-point), amplified by a 21-image val set where
one missed detection costs ~0.05 class-AP. All FP32-vs-INT8 comparisons use
the custom evaluator on both sides, so the comparison is internally exact.

## ONNX export & parity

`scripts/export_onnx.py`: opset 12, fixed batch 1, simplified.
Input `images [1,3,640,640]`, output `output0 [1,6,8400]`, 10.1 MB.
Deliverables: `models/best.pt`, `models/best.onnx`, `models/best_int8.onnx`.

**How parity was verified** (`scripts/verify_parity.py`, 10 val images, same
preprocessed tensor into PyTorch FP32-eval and ONNX Runtime; one shared
pixel pipeline in `scripts/common.py`):

- worst raw max abs diff: **8.85e-04** (tolerance 1e-3, script exits non-zero above)
- worst raw mean abs diff: **1.16e-05**
- worst post-NMS box coordinate difference: **6.1e-05 px**

## Quantisation — two real failures, diagnosed, then the trade-off

Static INT8 via ONX Runtime, QDQ, per-channel, MinMax, calibrated on **79
train-split images only** (`quantize.py` refuses val/test paths — calibrating
on eval images leaks evaluation data into the model).

Two genuine failures occurred and are kept in the history:
1. **Invalid graph**: per-channel `DequantizeLinear` needs the `axis`
   attribute (opset ≥13); quantising the opset-12 export directly fails.
   Fixed by upgrading a temporary copy to opset 13 for quantisation only.
2. **Total accuracy collapse (mAP 0.000)**: box coordinates survived but
   every class score quantised to exactly 0 — the YOLO head concatenates
   boxes (range 0–640) and sigmoid scores (0–1) into one tensor, and one
   per-tensor scale (~2.5) rounds all scores away. This is ANSWERS.md C1
   Cause 3 reproduced live. Fixed with Conv-only quantisation
   (`op_types_to_quantize=["Conv"]`).

### The three-way trade-off (val split; latency from `scripts/benchmark.py`)

| Model | Size (MB) | mAP@0.5 | mAP@0.5:0.95 | Mean (ms) | Median (ms) | p95 (ms) | Std (ms) |
|-------|-----------|---------|--------------|-----------|-------------|----------|----------|
| FP32 ONNX | 10.11 | 0.956 | 0.722 | 29.80 | 29.31 | 31.53 | 3.85 |
| INT8 ONNX | 3.00 | 0.933 | 0.689 | 64.40 | 63.42 | 70.47 | 7.80 |

Conditions: onnxruntime 1.29.0, CPUExecutionProvider, ORT default intra-op
threads, batch 1, 20 warmup + 200 timed iterations, laptop on mains with
light background load. **INT8 is 2.2× slower than FP32 on this CPU** — the
Q/DQ round-trips around each conv exceed the int8 compute saving on x86.
The −70% size and the −0.033 mAP@0.5:0.95 are real; the latency win is not,
on this hardware. Laptop latency under thermal variation is not exactly
reproducible; treat the distribution, not the mean, as the result. FP16
fallback was not needed (accuracy drop ≪ 0.15 gate).

## Held-out test — run exactly once

`scripts/evaluate_onnx.py --split test`, one run per model, no tuning after:

| Model | mAP@0.5 | mAP@0.5:0.95 |
|-------|---------|--------------|
| FP32 | 0.849 | 0.659 |
| INT8 | 0.793 | 0.609 |

**Val→test gap (FP32): −0.107 mAP@0.5, −0.063 mAP@0.5:0.95.** That is the
measured optimism of the val figures, and the single most informative line
here. The drop concentrates in `earphone_case` (0.915→0.779 AP50) — see the
failure analysis for why.

## Failure analysis (A4)

Full write-up with rendered prediction-vs-GT evidence:
[notes/failure_analysis.md](notes/failure_analysis.md)
(images in `notes/failure_examples/`). Summary: all three worst validation
images involve the **white earphone case** — classified as the (white)
charger at 0.60; missed entirely when adjacent to the white charger; and a
cross-class duplicate box surviving per-class NMS by design. The scene split
correctly kept white-case-on-glass scenes out of train, so the model met
that appearance thinly — the honest cost of a leak-free split at this scale.

## Reproduction

```bash
# 0. Python 3.11 venv. Install requirements FIRST, then CUDA torch LAST
#    (a requirements install replaced CUDA torch with a CPU build once —
#    always re-verify torch.cuda.is_available() after any pip operation):
pip install -r requirements.txt
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps

# 1. Dataset (images; labels + split manifest are in git)
python scripts/download_data.py

# 2. Rebuild from originals (only if starting from YoloData/)
python scripts/check_exif_orientation.py --images YoloData --csv YoloCSV.csv
python scripts/resize_images.py --src YoloData --dst data/prepared --long-edge 1280 --apply-exif
python scripts/cluster_scenes.py --images data/prepared --out data/scene_map.json --method colorhist --threshold auto
python scripts/split_dataset.py --images data/prepared --labels YoloLabels --scene-map data/scene_map.json --test-frac 0.33 --val-frac 0.2 --seed 42
python scripts/check_leakage.py --dataset data/dataset --report notes/leakage_report.md
python scripts/verify_labels.py --dataset data/dataset

# 3. Train (11m14s on an RTX 3050 Laptop)
python scripts/train.py --data data/data.yaml --model yolo11n.pt --imgsz 640 --batch 8 --epochs 100 --patience 25 --name final

# 4. Export + parity
python scripts/export_onnx.py --weights runs/detect/final/weights/best.pt --imgsz 640 --opset 12 --out models
python scripts/verify_parity.py --weights runs/detect/final/weights/best.pt --onnx models/best.onnx --images data/dataset/images/val --num-images 10

# 5. Eval cross-check, quantise, evaluate, benchmark
python scripts/evaluate_onnx.py --onnx models/best.onnx --split val
yolo val model=runs/detect/final/weights/best.pt data=data/data.resolved.yaml split=val
python scripts/quantize.py --onnx models/best.onnx --calib-images data/dataset/images/train --num-calib 79
python scripts/evaluate_onnx.py --onnx models/best_int8.onnx --split val
python scripts/benchmark.py --fp32 models/best.onnx --quant models/best_int8.onnx --image data/dataset/images/val/IMG_20260901_145029586_HDR.jpg --warmup 20 --iters 200

# 6. Held-out test (once)
python scripts/evaluate_onnx.py --onnx models/best.onnx --split test
python scripts/evaluate_onnx.py --onnx models/best_int8.onnx --split test

# 7. Failure-analysis renders
python scripts/render_predictions.py --onnx models/best.onnx --split val --out runs/render_val
```

## Assumptions

- YOLO txt is the single source of truth for labels; the CSV/XML exports are
  provenance only and stale after resize.
- Scene = "visual context" (background/lighting group from colour-histogram
  clustering), not "time block" — the large clusters span the whole session
  because the same backgrounds recur; the leakage audit, not the clustering,
  is the split's guarantee.
- The colorhist clustering threshold (0.10) came from the knee of an
  auditable scan; a different threshold changes cluster count (16 at 0.15,
  84 at 0.05) and therefore the split. The audit gates the outcome either
  way.
- Benchmarks are CPU (ONNX Runtime has no CUDA EP in this env by design —
  deployment for this exercise is CPU inference); training used the GPU.
- Evaluation P/R are reported at the best-F1 operating point of the IoU=0.5
  PR curve (Ultralytics' convention).

## Known gaps

1. **One 14-minute capture session, one lighting regime** — measured, not
   guessed: brightness std 21.8 over range 83–170, 10 background groups,
   box-area span 20.1× ([notes/variation_report.md](notes/variation_report.md)).
   Distance and background variation are adequate; lighting variation is
   essentially absent. A supplementary capture was recommended and declined
   for time; this is the dataset's central limitation and the first thing a
   deployment would expose.
2. **The annotation guide was finalised after labelling** (failed process
   gate). Mitigated by writing it descriptively from the labels and
   verifying 0 hard errors + full cross-format agreement; still the wrong
   order, and recorded as such.
3. **INT8 gives no latency win on this hardware** (2.2× slower; table
   above). On a deployment accelerator with native int8 paths the story
   likely inverts, but that is unmeasured here — see ANSWERS.md D5.
4. **Four physical objects only** (two per class). Intra-class variation
   exists (black + white of each) but generalisation to unseen units is
   unmeasured, and the failure analysis shows the white case is already the
   weak point.
5. **21-image val split** — one detection ≈ 0.05 class-AP; val numbers are
   coarse. The 50-image test set is the better estimate and is reported
   beside val everywhere.
6. Hard-image list (A2) was identified **post-hoc** from measured error,
   not declared at capture time.

## Confidence statement

Would these numbers hold on a different machine? The **accuracy** numbers:
yes within the stated coarseness — they are properties of the committed
weights and dataset, the split is auditable, and the pipeline is seeded and
reproducible end-to-end. The **latency** numbers: no — they are CPU-specific
(and laptop-thermal-specific), the INT8 result would likely invert on an
accelerator with native int8 execution, and I would re-run
`scripts/benchmark.py` on the target before quoting any figure there. The
honest uncertainty on accuracy is quantified by the val→test gap (−0.063
mAP@0.5:0.95) and bounded below by the white-case failure mode, which any
new environment will stress further.

## Recording

- Walkthrough video: TODO: add link (recorded after final push so on-screen
  numbers match this file).
