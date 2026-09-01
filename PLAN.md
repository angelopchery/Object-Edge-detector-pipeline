# PLAN.md — Artikate Assessment · Autonomous Execution Plan

**Supersedes all earlier versions of this file.** Written after the dataset landed and was audited.

---

## Operating contract

Claude Code executes this plan on my machine. It runs the commands, reads the real output, and writes measured values into the documents itself. I am not the executor.

**Claude Code has standing authority to:**
- Create and populate virtual environments, install packages
- Run training, export, quantisation, benchmark and evaluation to completion
- Write, refactor and delete scripts in `scripts/`
- Commit at every milestone
- Choose between documented alternatives using the decision tables in each phase

**Claude Code must stop and ask only when:**
- A decision table below has no matching branch and the choice materially affects what gets graded
- It would need to fabricate a number to proceed
- Something would be destructive to the raw data in `YoloData/`

Everything else: decide, act, document the decision in `notes/decisions.md`, keep going.

**Non-negotiable rules.**

1. No fabricated metrics. Every number in README.md is produced by a command Claude Code actually ran, and the producing command is named next to it.
2. Commit at every milestone. Never squash, never rewrite history.
3. Keep the `Co-Authored-By: Claude` trailer. The brief states AI tool use is expected.
4. A failed gate is a finding to write up, not an obstacle to route around. Failed gates go in Known Gaps.
5. Append to `notes/decisions.md` on every non-obvious choice: what was decided, what the alternatives were, why. This file is the raw material for the README's Assumptions section and for the live round.
6. Legible code over clever code. I have to modify this live while interviewers watch.

**Status key.** `[ ]` pending · `[~]` running · `[x]` done · `[!]` failed gate, documented

---

## What the data actually is

Established by audit, not assumption. Carry these facts forward; do not re-derive them.

| Property | Value |
|---|---|
| Images | 150 camera originals, 3072×4096 / 4096×3072 / 2448×3264 / 3264×2448, both orientations |
| Labels | 150 YOLO txt, 208 boxes. VOC XML and CSV agree box-for-box |
| Classes | `0 = EarphoneCase` (95 boxes), `1 = ChargingCase` (113 boxes) — **measured against the CSV, not assumed** |
| Co-occurrence | 92 images single-object, 58 images both |
| Box scale | Mostly 1–25% of frame, median ~5–10% |
| Capture window | One session, 14:50–15:04. **Fourteen minutes.** |
| Scene structure | No scene IDs in filenames. Timestamp clustering gives 8 clusters; one contains 87 images over ~4 continuous minutes |
| Verification | `verify_labels.py` passes: all coordinates in [0,1], no degenerate boxes, no orphans, class IDs only {0,1} |

**Three consequences that shape everything downstream.**

**Large boxes make mAP@0.5 uninformative.** With a median box at 5–10% of frame and few objects per image, IoU@0.5 is easy to clear. Expect a high mAP@0.5 that means little. **mAP@0.5:0.95 is the honest headline number** and the README should lead with it, saying why. This is the difference between a submission that reports 0.94 and one that explains what 0.94 does and does not establish.

**A fourteen-minute single-session capture is the dataset's central limitation.** The brief asked for deliberate variation in lighting, distance, angle and background. One continuous session in one place cannot have delivered much lighting or background variation. Phase 2 measures this rather than guessing, and the result drives a decision (see Phase 2 decision table). Whatever the outcome, this is the most important thing the README's Known Gaps section has to say, and reporting it plainly scores better than any marginal mAP gain.

**Filename-based scene splitting is dead and hand-partitioning 87 images is not a real option.** Phase 3 derives scene identity from visual similarity instead, then *audits* the resulting split for leakage. That is both automatic and more defensible than a naming convention, and the audit is the actual guarantee — the clustering is just the means.

---

## Phase 0 — One-time decision batch  `[~]`  _(batch prepared, awaiting answers)_

Everything I need to answer, asked once, up front. Claude Code prepares the material, I answer in a single message, and then autonomy runs to Phase 11.

Claude Code prepares:

1. **Class semantics.** Crop one labelled instance of each class to `notes/class_check/class0_earphonecase.jpg` and `class1_chargingcase.jpg`. I confirm which physical object is which in one line. Do not assume from the name — "ChargingCase" is ambiguous between the brick and a case.

2. **Capture variation report.** Run the Phase 2 audit first and present the numbers (brightness spread, background diversity proxy, orientation mix). I answer whether a second 20-minute capture session under different lighting is worth doing, informed by real numbers rather than a guess.

3. **A drafted `ANNOTATION_GUIDE.md`, written descriptively from evidence.** Claude Code inspects the labels and infers the rules that were actually applied — are edge-cropped objects boxed? Is an open case one box or two? Are there boxes on heavily occluded instances? — then writes the guide as a description of observed practice with the inferred rule stated and marked `[inferred]`. I correct anything wrong in one pass. This replaces me writing seven rules from memory.

4. **Hard-image candidates.** After Phase 4 training, Claude Code ranks validation images by error and proposes which qualify as the "10 images you expect the model to find hard." I confirm or amend. The README states plainly that these were identified post-hoc rather than declared at capture time — that is honest and cheap; pretending otherwise is neither.

Bundle all four into one message. Do not ask them serially.

**Commit:** `docs: phase 0 decision batch prepared`

---

## Phase 1 — Environment  `[x]`  _(gate passed: torch 2.5.1+cu121 on RTX 3050; two env defects caught, see decisions.md)_

Claude Code executes. The current venv is Python 3.14; CUDA torch and onnxruntime may not publish wheels for it.

```bash
py -3.11 -m venv artikate
artikate\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch, onnxruntime as ort; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print('ort', ort.__version__, ort.get_available_providers())"
```

**Decision table.**

| Condition | Action |
|---|---|
| CUDA available, RTX 3050 named | Proceed |
| `cuda.is_available()` False | CPU build installed — reinstall from the cu121 index. Retry once |
| Still False after retry | Proceed on CPU. Reduce epochs to 60, note the change, and record in Known Gaps that training and all latency figures are CPU-bound |
| onnxruntime has no 3.11 wheel | Try 3.12. If that also fails, record and use whatever ORT version installs |

Write the exact resolved versions of torch, ultralytics, onnx, onnxruntime, numpy and opencv into `notes/environment.md` — the README's hardware section quotes that file.

**Gate.** The one-liner runs and its output is captured verbatim into `notes/environment.md`.

**Commit:** `chore: python 3.11 environment, versions recorded`

---

## Phase 2 — Data preparation and variation audit  `[x]`  _(exif gate 150/150, resize 150/150, variation audited — brightness borderline-narrow)_

### 2a. EXIF orientation gate

makesense.ai labelled the images as the browser rendered them, with EXIF rotation applied. If PIL's raw dimensions disagree with the CSV's recorded `image_width`/`image_height`, the normalised labels and the pixels are in different orientations and nothing downstream is valid.

```bash
python scripts/check_exif_orientation.py --images YoloData --csv YoloCSV.csv
```

For every image, compare `ImageOps.exif_transpose(Image.open(p)).size` against the CSV dimensions.

| Condition | Action |
|---|---|
| All match | Proceed. Record that the check was run and passed |
| Mismatches present | Apply `exif_transpose` during resize so pixels match the labelling orientation. Re-verify. This is a caught defect — write it up |
| Mismatch persists after transpose | **Stop and ask.** Nothing can train correctly through an orientation error |

### 2b. Resize

```bash
python scripts/resize_images.py --src YoloData --dst data/prepared --long-edge 1280 --apply-exif
```

YOLO txt labels are normalised, so resizing after labelling is safe for them. The VOC XML and CSV carry absolute pixel coordinates and go stale — they stay in the repo as annotation provenance only. **YOLO txt is the single source of truth downstream.** State this explicitly in the README so a reader does not assume all three formats are live.

**Gate.** Output count is 150. Three portrait images spot-checked for correct orientation.

### 2c. Variation audit

This measures the fourteen-minute-session concern instead of speculating about it.

```bash
python scripts/audit_variation.py --images data/prepared --labels YoloLabels --out notes/variation_report.md
```

Compute and report:
- Mean/std/min/max of image brightness, and a histogram — is there real lighting variation or one lighting condition?
- Background diversity proxy: colour histogram of the region outside all boxes, clustered; how many distinct backgrounds are there really?
- Box-area distribution as a fraction of frame — is there genuine distance variation, or was everything shot from one distance?
- Orientation mix, aspect-ratio mix
- Per-class instance counts, and boxes per image

| Condition | Action |
|---|---|
| Brightness std is wide, ≥3 background clusters, box areas span an order of magnitude | Good variation. Proceed, quote the numbers in the README as evidence |
| Narrow brightness, ≤2 backgrounds, or box areas clustered in under a 3× range | Surface in the Phase 0 batch with the numbers. Recommend a 20-minute supplementary capture — different room, different light, a few extreme distances — as the single highest-value use of remaining time. If I decline, proceed and make this the lead item in Known Gaps, quantified |

**Commit:** `data: resize to 1280px, exif verified, variation audited`

---

## Phase 3 — Scene derivation, split, and leakage audit  `[~]`  _(3a done: colorhist, 40 clusters; 3b/3c after Phase 0 answers)_

The most important phase. Filename-based scenes do not exist and timestamp clustering cannot separate an 87-image continuous burst. Derive scene identity visually, then verify the split empirically.

### 3a. Visual scene clustering

```bash
python scripts/cluster_scenes.py --images data/prepared --out data/scene_map.json --method phash --threshold auto
```

- Compute a perceptual hash (and, if available, a small CNN embedding) for all 150 images
- Agglomerative clustering on pairwise distance
- Choose the threshold by scanning the range and taking the knee of the cluster-count curve; write the scan to `notes/scene_clustering.md` so the choice is auditable
- Timestamps are used as a weak prior, not the primary signal

| Condition | Action |
|---|---|
| 15–40 clusters | Healthy. Proceed |
| Under 10 clusters | The 150 images contain very few distinct arrangements. Proceed, but this is a **major finding**: effective dataset size is far smaller than the image count implies. Lead with it in Known Gaps and connect it to the failure analysis |
| Over 60 clusters | Threshold too tight; near-duplicates are being separated and the split will leak. Loosen and re-run |

Supersedes the timestamp-derived `scene_map.json` and its `reviewed: true` requirement — the leakage audit in 3c is a stronger guarantee than manual review, and it is reproducible.

### 3b. Split

Hold back whole clusters, never individual images. Target ~50 images as an untouched test set, remainder split ~80/20 train/val.

```bash
python scripts/split_dataset.py --images data/prepared --labels YoloLabels \
    --scene-map data/scene_map.json --test-frac 0.33 --val-frac 0.2 --seed 42
```

Assign clusters to splits greedily to balance both image count and per-class instance count.

### 3c. Leakage audit — this is the real gate

```bash
python scripts/check_leakage.py --dataset data/dataset --report notes/leakage_report.md
```

Compare every val and test image against every train image by perceptual hash distance. Report the closest pairs with distances.

| Condition | Action |
|---|---|
| No cross-split pair below the near-duplicate threshold | Split is clean. Commit the report as evidence |
| Violations found | Move the offending cluster wholesale to the training side, re-split, re-audit. Iterate until clean |
| Cannot reach clean without emptying a split | Report honestly: the dataset does not support a clean split at this size. Proceed with the best achievable, quantify the residual leakage, and state in the README that validation figures are optimistic by an amount the audit measures |

**Gate — all four:**
1. Leakage audit clean, or residual quantified and documented
2. Both classes present in train, val and test in roughly source proportion
3. `data/split_manifest.json` and `notes/leakage_report.md` committed
4. No cluster spans two splits

Do not resample to fix class imbalance. Record it and connect it to the failure analysis.

**Commit:** `data: visual scene clustering, scene-aware split, leakage audit clean`

---

## Phase 4 — Training  `[ ]`

```bash
python scripts/train.py --data data/data.yaml --model yolo11n.pt --imgsz 640 --batch 8 --epochs 100 --patience 25 --name final
```

**OOM ladder — apply automatically, no need to ask:** batch 8 → 4 → 2 → `imgsz 512`. Record whatever actually ran; `resolved_config.json` captures it and the README quotes that file, not the intended config.

Capture wall-clock training time.

**Gate — plausibility. Read mAP@0.5:0.95, not mAP@0.5.** Given the box sizes, mAP@0.5 will be high and is not the discriminator.

| Condition | Action |
|---|---|
| mAP@0.5:0.95 in 0.45–0.80 | Plausible. Proceed |
| mAP@0.5:0.95 above 0.85, or mAP@0.5 above 0.97 | Re-run `check_leakage.py` at a tighter threshold. If it stays clean, the result is real but the dataset is easy — say so explicitly and attribute it to large boxes, two visually distinct rigid objects, and a single capture session. Do not report a high number without this explanation |
| mAP@0.5:0.95 below 0.30 | Diagnostic ladder, in order: (1) `render_labels.py` on train — are boxes on the right objects with the right classes? (2) confirm `data.yaml` names match measured IDs 0=EarphoneCase, 1=ChargingCase; (3) confirm pretrained weights loaded rather than training from scratch; (4) check train/val class balance. Fix, retrain, **and keep both runs in the history** |

A run that went wrong and was diagnosed is explicitly something the graders want to see. Do not clean it up.

**Commit:** `train: yolo11n <N> epochs, mAP@0.5:0.95 = <measured>`

---

## Phase 5 — ONNX export and parity  `[ ]`

```bash
python scripts/export_onnx.py --weights runs/detect/final/weights/best.pt --imgsz 640 --opset 12 --out models
python scripts/verify_parity.py --weights runs/detect/final/weights/best.pt --onnx models/best.onnx --images data/dataset/images/val --num-images 10
```

The brief asks *how* parity was confirmed. Report actual numbers: max abs diff and mean abs diff on raw output tensors, plus worst-case post-NMS box disagreement in pixels.

| Condition | Action |
|---|---|
| Max abs diff < 1e-3 and boxes agree within 1 px | Proceed |
| Diff between 1e-3 and 1e-2 | Investigate but do not block. Common cause is fp32 accumulation order. Report the number and the diagnosis |
| Diff > 1e-2 or boxes disagree by > 1 px | Preprocessing mismatch. Verify both paths import `scripts/common.py`. Check normalisation, channel order, letterbox pad value, and dynamic-shape handling. This is a real defect — fix, then write it up as a caught bug |

**Commit:** `export: onnx opset 12, parity max abs diff <measured>`

---

## Phase 6 — Evaluation cross-check  `[ ]`

Before quantising anything. `evaluate_onnx.py` hand-rolls AP; if it disagrees with Ultralytics, every downstream comparison is meaningless.

```bash
python scripts/evaluate_onnx.py --onnx models/best.onnx --split val
yolo val model=runs/detect/final/weights/best.pt data=data/data.yaml split=val
```

| Condition | Action |
|---|---|
| mAP@0.5 figures agree within 0.01 | Proceed. Record both numbers in the README — the cross-check itself is worth points |
| Disagree by 0.01–0.05 | Likely a convention difference (P/R operating point, AP interpolation). Document precisely which, then proceed |
| Disagree by more than 0.05 | The custom implementation is the suspect. Fix it against Ultralytics as reference before it contaminates the quantisation comparison |

**Commit:** `eval: custom mAP cross-validated against ultralytics`

---

## Phase 7 — Quantisation and benchmark  `[ ]`

```bash
python scripts/quantize.py --onnx models/best.onnx --calib-images data/dataset/images/train --num-calib 100
python scripts/benchmark.py --fp32 models/best.onnx --quant models/best_int8.onnx --image <val_image> --warmup 20 --iters 200
python scripts/evaluate_onnx.py --onnx models/best_int8.onnx --split val
```

Calibration draws only from train. Keep the guard that refuses paths containing `val` or `test`, and keep its comment.

| Condition | Action |
|---|---|
| Static INT8 succeeds, mAP drop under 0.15 | Report the trade-off table |
| INT8 mAP drop above 0.15 | Report it as measured — a large drop is a legitimate finding and connects directly to the Part C1 answer. Also produce FP16 and report all three |
| Static INT8 fails or is unsupported | Fall back to FP16, state which and why. The brief explicitly permits this |

Benchmark conditions to record alongside the numbers: execution provider, thread count, mains vs battery, other load. Laptop latency under thermal throttling is not reproducible and saying so is the honest answer to the confidence question.

If accuracy dropped, quantify it as a number.

**Commit:** `bench: fp32 vs int8 latency, size, accuracy`

---

## Phase 8 — Held-out test, run exactly once  `[ ]`

```bash
python scripts/evaluate_onnx.py --onnx models/best.onnx --split test
```

One run. No tuning afterwards, under any circumstances, including if the number is disappointing.

Report it beside the validation number and comment on the gap. **The val-to-test gap is the single most informative line in the README** — it quantifies exactly how optimistic the validation figure was, and it is the direct answer to "how confident are you these numbers would hold on a different machine."

**Commit:** `eval: single-shot held-out test, no tuning after`

---

## Phase 9 — Failure analysis (A4)  `[ ]`

```bash
python scripts/render_predictions.py --onnx models/best.onnx --split val --out runs/render_val
```

Filenames prefix `errNN_` by combined missed-GT and false-positive count, so the worst sort to the top.

Claude Code writes the analysis from the rendered evidence. For each of the three worst: predicted class/box/confidence, what it should have been, a **specific** hypothesis, and what to capture or change.

Hypotheses must connect to measured facts from this repo, not generic causes. Available and relevant:
- Single 14-minute capture session — the variation report quantifies exactly how narrow
- Cluster count from Phase 3 — how many distinct arrangements really exist
- Per-class imbalance, 95 vs 113
- Box-size distribution — whether small-object cases are under-represented
- Number of distinct physical units photographed; if there is only one charger and one case, say so, and state that validation overstates generalisation to unseen units

**Commit:** `docs: part A4 failure analysis with rendered evidence`

---

## Phase 10 — Parts C and D  `[ ]`

No data dependency. Draft during Phase 4 training rather than waiting.

**Part C** — the rubric grades elimination logic, not conclusions. Each answer: what is checked first and in what order, what is ruled out and on what evidence, root-cause hypothesis, fix, and validation before it reaches the client.

- **C1** — quantisation collapse 0.91 → 0.58. At least three independent root causes, each with a distinguishing test. Candidates: calibration set unrepresentative of the deployment distribution; per-tensor rather than per-channel quantisation on a depthwise-separable backbone; preprocessing divergence between the PyTorch and TensorRT paths; unfused or unsupported ops silently falling back; a detection head whose output range does not survive INT8. **If Phase 7 measured a real INT8 drop, cite my own measured number here** — a diagnosis grounded in the candidate's own benchmark is far stronger than a textbook list.
- **C2** — one camera of twelve, consistent directional offset, worse at frame edges. The pattern localises the defect to the per-stream coordinate transform, not the model. State the reasoning: a model defect would affect all twelve; a random defect would not be directional; an offset that grows toward the edges is a scale error, one that is constant is a translation error, and both together is the letterbox signature. **Reference the Snippet 1 answer explicitly** — same defect class, and connecting them shows the diagnosis generalises. Confirming without physical access: pull one frame from the stream, compare the reported resolution against the other eleven, project a known fixed reference point.
- **C3** — 97% → 84% over three months, nothing reported changed. Two or three causes with confirming evidence in logs or images. Then a monitoring signal with a **specific threshold and a specific measurement**, not "monitor for drift."

**Part D** — arithmetic where arithmetic is possible.

- **D1** — 8 × 15 = 120 fps aggregate. Show it. Then separate the two budgets explicitly: throughput requires ≥120 inferences/sec sustained; latency requires each individual frame to complete within 200 ms end to end including decode, preprocess, inference, postprocess and downstream action. Batching helps throughput and *hurts* per-frame latency. Conflating them is the standard error and calling it out is worth points.
- **D2** — model family and precision, why the combination fits, and the single first measurement to take. Prefer "I would benchmark X before committing" to an unverified confident number; the brief asks for exactly that.
- **D3** — air-gap retraining loop: how an operator flags a detection at the line, how feedback physically crosses the gap, how a candidate model is validated before it replaces the running one.
- **D4** — rollback: detection latency, trigger signal, and the mechanism.
- **D5** — genuinely least confident part and what would resolve it. Answer honestly; the brief says a candid answer scores better than false confidence.

**Commits:** `docs: part C production failure diagnoses`, then `docs: part D edge deployment design`

---

## Phase 11 — README  `[ ]`

Every `TODO: measure` replaced with a real value, each with its producing command named.

Required sections:
- Hardware and software versions, quoted from `notes/environment.md`
- Dataset: per-class counts, split counts, capture methodology **including the 14-minute single-session limitation stated plainly**, cluster count, and the scene-derivation rationale
- Annotation: link to the guide, the actual workflow (all 150 hand-labelled in one pass — no seed model, no prelabel loop), and the fact that the guide was written descriptively after labelling
- The class-ID defect that was caught and fixed, with how it was measured. **This is a strong story — a silent defect found by cross-referencing formats rather than by testing.** Give it real space
- Training config from `resolved_config.json`
- Validation, held-out test, and the gap between them, with mAP@0.5:0.95 leading and an explanation of why mAP@0.5 is not the discriminator here
- ONNX parity: the numbers and the method
- Three-way trade-off table
- Exact reproduction commands, in order
- **Assumptions** — built from `notes/decisions.md`
- **Known gaps** — the capture-variation limitation first, then the annotation-guide ordering, then anything cut
- **Confidence statement** — would these numbers hold elsewhere, grounded in the val-to-test gap and the benchmark conditions
- Recording link

**Gate.** Someone unfamiliar with the repo can reproduce every number from the README alone.

**Commit:** `docs: complete README with measured results`

---

## Phase 12 — Recording and final audit  `[ ]`

The only phase requiring me.

Record 6–8 minutes after the final push, so on-screen numbers match the README:
1. Dataset and annotations. **Point out one genuinely ambiguous label**, and reference the guide rule that resolved it
2. ONNX inference on a validation image, live
3. `benchmark.py` running, numbers appearing
4. The Part A decision I am least sure about — the scene-clustering threshold and what it means for the split is the honest choice here

Claude Code then runs the final audit:

- [ ] `git log --oneline` shows the real sequence including failed gates and fixes
- [ ] No number in README lacks a producing command
- [ ] Weights and ONNX committed, or linked with SHA256
- [ ] `download_data.py` works from a clean clone and the checksum verifies
- [ ] ANSWERS.md has all nine sections (B1–B3, C1–C3, D1–D5)
- [ ] Assumptions and Known Gaps both non-empty and substantive
- [ ] Recording link resolves
- [ ] Repository public or shared with the sending address

**Commit:** `docs: recording link and final audit`

---

## If time runs short

Priority order from the brief:

1. Phases 1–4 — dataset and a model that trained, with honest numbers
2. Phase 10 Part C — with Part B already done, these carry 40% between them
3. Phases 5–7 — export and benchmark, even FP32 vs FP16 on CPU
4. Phases 8–9 and Part D

Then write down precisely what was cut. The brief says that section is read closely.

---

## Live round preparation

Not a deliverable, but where this is won. Before the call I must be able to, unaided:

- Add a third class to `data.yaml` and start a run
- Change an augmentation and explain the expected effect
- Move the confidence threshold and explain why precision and recall move in opposite directions
- Walk the commit history and explain each commit
- Defend one Part D decision under pushback — pick the weakest in advance and prepare the honest concession rather than a defence

Claude Code should maintain `notes/walkthrough.md`: one line per script explaining what it does and the one thing about it I would be asked. Anything in this repo I could not modify under observation is a liability.