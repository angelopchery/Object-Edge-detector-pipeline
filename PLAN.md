# PLAN.md — Artikate Assessment Execution Plan

**Purpose.** This is the standing plan for the remainder of the assignment. Claude Code should read this file at the start of every session and work from it rather than waiting for fresh instructions. Update the status markers as stages complete and commit those updates — the file doubles as a progress log.

**Roles.**
- Claude Code writes scripts, fills documents, and commits. It never invents a measured number.
- I run every script, capture the real output, and paste it back.
- Any number appearing in README.md must be traceable to a command in this plan that I actually executed.

**Status key.** `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

---

## 0. Standing rules

1. **No fabricated metrics, ever.** If a value is not yet measured, it stays as `TODO: measure (produced by scripts/X.py)`. A plausible-looking placeholder that survives into the final README is worse than an empty slot, because it will contradict the screen recording.
2. **Commit at every milestone**, with a message describing what changed and why. Never squash, never rebase history. The commit graph is graded evidence of process.
3. **Keep the `Co-Authored-By: Claude` trailer.** The brief states outright that AI tool use is expected. Stripping it is the only dishonest act available in this repo.
4. **Every stage below has a gate.** Do not proceed past a gate that failed. A failed gate is a finding to write up, not an obstacle to route around.
5. **Log surprises in `notes/capture_log.md` as they happen.** Anything that surprised me during the run is material for the README's Assumptions section and for the live round.
6. **Prefer legible code over clever code.** I will be editing this live on a call while interviewers watch.

---

## 1. Environment  `[~]`  _(gitignore/models fix done in 52f13c7; venv rebuild + GPU gate pending)_

The current venv is Python 3.14; onnxruntime and CUDA-enabled torch may not publish wheels for it. Rebuild before losing time to wheel errors.

```bash
# Windows
py -3.11 -m venv artikate
artikate\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then verify the stack actually sees the GPU:

```bash
python -c "import torch, onnxruntime as ort; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print(ort.__version__, ort.get_available_providers())"
```

**Gate.** `torch.cuda.is_available()` is `True` and reports the RTX 3050. If it is `False`, the installed torch is the CPU build — reinstall from the CUDA index before continuing. Record the exact torch, ultralytics, onnx and onnxruntime versions into README's hardware section.

**Also fix now, before the first data commit:**
- `.gitignore` currently excludes `*.pt` and `*.onnx` wholesale. YOLO11n weights are ~5 MB and the ONNX ~10 MB, both well inside GitHub limits, and deliverable A5 asks for them. Un-ignore `models/best.pt`, `models/*.onnx`. Keep `runs/` ignored.
- Confirm `data/data.yaml` lists `names: [charger_brick, earphone_case]` in an order matching the class IDs makesense.ai actually exported. Open one label `.txt` alongside its image and confirm by eye that class `0` is on the brick. Flipped class IDs train cleanly and produce plausible metrics — this is exactly the class of silent defect the assignment is testing for.

**Commit:** `chore: pin python 3.11 environment, correct gitignore for deliverable weights`

---

> **Reality check (2026-09-01, after data landed).** The dataset arrived
> differently from what stages 2–5 assumed, and the plan below is amended
> accordingly:
> - All 150 images are already hand-labelled (makesense.ai; YOLO txt + VOC
>   XML + CSV all agree, 208 boxes, `verify_labels.py` passes with 0 errors).
>   The seed-model/pre-label loop (stages 4–5) never happened — the README
>   must describe the actual workflow, not the planned one.
> - Filenames are camera originals, not `sceneNN_x.jpg`. Scene identity now
>   lives in `data/scene_map.json` (drafted from timestamp gaps by
>   `build_scene_map.py`, must be human-reviewed before the split will
>   accept it).
> - Images are full-resolution (3072×4096 etc.), not yet resized. Resizing
>   after labelling is safe for the YOLO txt labels because they are
>   normalised; the CSV/XML exports carry absolute pixels and become
>   stale after resize — they stay in the repo as annotation provenance
>   only, and YOLO txt is the single source of truth downstream.
> - The class IDs in the export were the reverse of the scaffolded
>   `data.yaml` (measured: 0=EarphoneCase, 1=ChargingCase). Fixed in commit
>   7324505 — this belongs in the README as a caught-defect story.
> - Annotation happened before ANNOTATION_GUIDE.md was finalised — a failed
>   stage-3 gate. Per rule 4 that is a finding to report: the guide gets
>   written descriptively (the rules actually applied) and the deviation
>   goes in Known Gaps.

## 2. Image preparation  `[~]`  _(labelling done first; resize + scene review pending)_

```bash
python scripts/resize_images.py --src YoloData --dst data/prepared --long-edge 1280
```

Confirm the output count matches the input count and that EXIF rotation was applied (portrait photos must not come out sideways — check three by eye).

**EXIF/label-orientation gate (new).** makesense.ai labelled the images as the
browser displayed them (EXIF applied). After the env rebuild, verify for a few
portrait images that PIL's `exif_transpose` dimensions match the `image_width`/
`image_height` recorded in YoloCSV.csv for that file. If they disagree, the
normalised labels and the resized pixels are in different orientations — stop
and resolve before anything trains.

**Scene map review (new, replaces the sceneNN filename assumption).**
`data/scene_map.json` is drafted (8 scenes from timestamp gaps). Three
clusters are flagged as spanning multiple physical arrangements and must be
split by hand by the person who took the photos: scene03 (15 images),
scene05 (16), scene08 (87 — nearly 4 minutes of continuous shooting).
Split them, set `"reviewed": true`, commit. `split_dataset.py` refuses an
unreviewed map.

**Reserve the held-out test set now, before any annotation or training.** From ~150 images, hold back ~50 as a test set touched exactly once at the very end. The remaining ~100 become train/val. This exceeds what the brief asks for and gives the strongest possible answer to "how confident are you these numbers hold on a different machine."

Hold back **whole scenes**, not individual images. Record them as JSON, e.g.:

```json
{"test_scenes": ["scene03", "scene07", "scene11", "..."]}
```

`split_dataset.py --test-scenes data/heldout_scenes.json` (stage 6) will force
exactly these scenes into the test split and shuffle only the remainder.

**Gate.** Test scene IDs are recorded in `data/heldout_scenes.json` and committed. Nothing downstream reads that folder until stage 12.

**Commit:** `data: resize to 1280px long edge, reserve held-out test scenes`

---

## 3. Annotation standard  `[!]`  _(gate failed: labelling happened first — write the guide descriptively, record in Known Gaps)_

Finalise `ANNOTATION_GUIDE.md` before labelling anything. Replace the placeholders with decisions. At minimum:

- Occlusion threshold — below what visible fraction is an instance skipped?
- Frame-edge crops — box the visible portion, or exclude?
- An **open** earphone case — one box around the whole thing, or separate lid and body? Decide once.
- A brick plugged into a wall socket — box the visible face only, or the implied full body?
- Minimum box size in pixels below which an instance is skipped.
- Out-of-focus background instances — labelled or not?
- Box tightness convention: tight to the visible silhouette, excluding shadow.

**Gate.** The guide is committed *before* the first label file. The commit order matters and is visible in the history.

**Commit:** `docs: finalise annotation guideline before labelling`

---

## 4. Seed annotation — 40 images  `[x]`  _(superseded: all 150 hand-labelled directly; verify_labels passed with 0 errors — commit f5230ab)_

Hand-label ~40 images in makesense.ai spanning easy, cluttered, dark and far-away conditions. Export YOLO txt.

```bash
# flat working folders during annotation: images in data/prepared, labels in data/labels_all
python scripts/verify_labels.py --images data/prepared --labels data/labels_all
```

(Only the annotated subset should be in the checked folders at this point —
an unannotated image would correctly be reported as an orphan.)

**Gate.** Zero orphans, zero out-of-range coordinates, zero invalid class IDs, no zero-area boxes. Review anything flagged as suspiciously small.

Then render every box and scroll through the output — the check that catches
what the numeric one cannot (box on the wrong object, wrong class, loose box):

```bash
python scripts/render_labels.py --images data/prepared --labels data/labels_all --out runs/render_seed
```

Record per-class instance counts from the script output into `notes/capture_log.md`.

**Commit:** `data: hand-labelled 40 seed images, verified`

---

## 5. Seed model and pre-labelling  `[x]`  _(superseded: no prelabel loop was used — README must say so; prelabel.py remains for future data)_

```bash
python scripts/train.py --data data/data.yaml --epochs 50 --name seed
python scripts/prelabel.py --weights runs/detect/seed/weights/best.pt --source data/prepared_unlabelled
# label txts land in runs/detect/prelabel/labels/, annotated previews alongside
```

(Note: the seed model trains on the 40 seed images split scene-aware into a
temporary train/val — do not train on all 40 with no val, and do not let seed
val scenes come from the held-out test scenes.)

Import images plus predicted labels into makesense.ai and **review every box**. Expect to correct roughly a third and delete some false positives. This review is not optional — it is the part of the workflow that makes model-assisted labelling legitimate rather than a shortcut, and it is what I will be asked about.

Note in `notes/capture_log.md`: how many boxes were corrected, how many deleted, how many added. That ratio is a genuinely interesting number to report.

Re-run `verify_labels.py` on the full set.

**Commit two separate commits:** `data: bootstrap labels from 40-image seed model` then `data: manual review and correction of pre-labelled boxes`. The separation makes the workflow legible in the history.

---

## 6. Split  `[ ]`

```bash
python scripts/split_dataset.py --images data/prepared --labels YoloLabels \
    --scene-map data/scene_map.json --test-scenes data/heldout_scenes.json --seed 42
python scripts/verify_labels.py --dataset data/dataset
```

(Requires `data/scene_map.json` with `"reviewed": true` — stage 2.)

**Gate — all four must hold:**
1. The disjointness assertion on scene sets passes.
2. Both classes are present in both train and val, in roughly the source proportion.
3. `data/split_manifest.json` is written and committed.
4. No scene ID appears in both splits **and** no held-out test scene appears in either.

If one class is badly under-represented, do not resample to fix it — record the imbalance and connect it to the A4 failure analysis. Honest imbalance reported is worth more than imbalance quietly corrected.

**Commit:** `data: scene-aware train/val split with committed manifest`

---

## 7. Training  `[ ]`

```bash
python scripts/train.py --data data/data.yaml --model yolo11n.pt --imgsz 640 --batch 8 --epochs 100 --patience 25 --name final
```

If CUDA runs out of memory: drop batch to 4, then to 2. If it still fails, drop `imgsz` to 512. Record whatever was actually used — `resolved_config.json` captures this automatically, and the README must quote that file rather than the intended config.

Record wall-clock training time.

**Gate — plausibility, and this is the one that matters most.**

The brief states plainly that near-perfect results on ~80 self-captured images will be read as evidence of a leaked split, not as strength.

- **mAP@0.5 in roughly 0.70–0.90** → plausible for two distinctive rigid objects on a clean scene-aware split. Proceed.
- **mAP@0.5 above 0.95** → stop and investigate before reporting anything. Check: are val images near-duplicates of train images despite the scene split (did I photograph the same arrangement across two scene numbers)? Did any pre-labelled image get copied into both folders? Run `python scripts/check_leakage.py --dataset data/dataset` — it perceptual-hashes every val/test image against every train image and reports the closest pairs.
- **mAP@0.5 below 0.55** → check class IDs are not flipped, check the labels render on the objects (`python scripts/render_labels.py --images data/dataset/images/train --labels data/dataset/labels/train --out runs/render_train`), check the model is not training from scratch instead of pretrained weights.

Whatever happens here, write it down. A run that went wrong and was diagnosed is explicitly listed as something the graders want to see in the commit history.

**Commit:** `train: yolo11n final run, <N> epochs, mAP@0.5 = <measured>`

---

## 8. ONNX export and parity  `[ ]`

```bash
python scripts/export_onnx.py --weights runs/detect/final/weights/best.pt --imgsz 640 --opset 12 --out models
python scripts/verify_parity.py --weights runs/detect/final/weights/best.pt --onnx models/best.onnx --images data/dataset/images/val --num-images 10
```

**Gate.** Max absolute difference on raw output tensors below `1e-3` across all ten images, and final post-NMS box coordinates agreeing within 1 px.

The brief asks *how* parity was confirmed, not that it was. The README must state the actual numbers — max abs diff, mean abs diff, worst-case pixel disagreement — not the sentence "outputs matched."

If parity fails: the usual causes are a preprocessing mismatch between the two paths (different normalisation, BGR vs RGB, different letterbox pad value) or dynamic-shape handling. `scripts/common.py` exists specifically so both paths share one pixel pipeline — verify both are actually importing it.

**Commit:** `export: onnx opset 12, parity verified to <measured> max abs diff`

---

## 9. Evaluation cross-check  `[ ]`

**Do this before quantising anything.** `evaluate_onnx.py` hand-rolls AP computation. If it disagrees with Ultralytics, the FP32 and INT8 numbers are not comparable and at least one of them is meaningless.

```bash
python scripts/evaluate_onnx.py --onnx models/best.onnx --split val
# compare against Ultralytics' own val metrics on the identical split
yolo val model=runs/detect/final/weights/best.pt data=data/data.yaml split=val
```

**Gate.** The two mAP@0.5 figures agree within ~0.01. If they diverge materially, the custom implementation is the suspect, not the model — fix it before it contaminates every downstream comparison. Note in the README that this cross-check was performed and what the two numbers were; it is exactly the kind of verification the rubric rewards.

**Commit:** `eval: cross-validate custom mAP against ultralytics on identical split`

---

## 10. Quantisation  `[ ]`

```bash
python scripts/quantize.py --onnx models/best.onnx --calib-images data/dataset/images/train --num-calib 100
# INT8 is the default mode; the FP16 fallback is: python scripts/quantize.py --onnx models/best.onnx --fp16
```

Calibration draws **only** from the training split. The script already hard-refuses paths containing `val` or `test`; keep that guard and keep its comment.

If static INT8 fails or degrades catastrophically, fall back to FP16 and **state which and why** — the brief explicitly allows this and asks for the justification.

Record all three file sizes in MB.

**Commit:** `quantize: static int8 via ORT, calibration from train split only`

---

## 11. Benchmark  `[ ]`

```bash
python scripts/benchmark.py --fp32 models/best.onnx --quant models/best_int8.onnx \
    --image data/dataset/images/val/<any_val_image>.jpg --warmup 20 --iters 200
# batch size is fixed at 1 inside the script (the export is fixed-batch-1 anyway)
python scripts/evaluate_onnx.py --onnx models/best_int8.onnx --split val
```

The README table needs all three axes, per the brief:

| Model | Size (MB) | Latency mean (ms) | Latency p95 (ms) | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| FP32 ONNX | | | | | |
| INT8 ONNX | | | | | |

Also record: execution provider, thread count, whether the laptop was on mains or battery, and whether anything else was running. Latency on a laptop under thermal throttling is not reproducible, and saying so is the honest answer to "how confident are you these numbers hold on a different machine."

If accuracy dropped, **quantify the drop as a number**, not a description.

**Commit:** `bench: fp32 vs int8 latency, size and accuracy on measured hardware`

---

## 12. Held-out test set — run exactly once  `[ ]`

```bash
python scripts/evaluate_onnx.py --onnx models/best.onnx --split test
```

Run it once. Do not tune anything afterwards. Report the number whatever it is, alongside the val number, and comment on the gap. A test score materially below val is the single most informative line in the whole README, because it quantifies exactly how much the val figure was optimistic.

**Commit:** `eval: single-shot held-out test evaluation, no tuning after`

---

## 13. Part A4 — failure analysis  `[ ]`

Pick the three worst validation images by per-image loss or by manual inspection of predictions:

```bash
python scripts/render_predictions.py --onnx models/best.onnx --split val --out runs/render_val
```

Output filenames are prefixed `errNN_` (missed GT + false positives per image),
so the worst images sort to the top. For each of the three worst, write:

- What the model predicted (class, box, confidence).
- What it should have predicted.
- A **specific** hypothesis: annotation inconsistency, class imbalance, scale, occlusion, glare, insufficient examples of that condition.
- What I would capture or change to fix it.

Include the rendered prediction-vs-ground-truth images in the repo.

Cross-reference honestly: if the dataset contains only two physical charger units, say so here and explain that the val figure likely overstates generalisation to unseen units. That single sentence is worth more than a marginal mAP improvement.

**Commit:** `docs: part A4 failure analysis with rendered examples`

---

## 14. Parts C and D  `[ ]`

Written answers, no code, no data dependency. Can be drafted in parallel with any of the above while training runs.

**Part C** — three production diagnoses. The rubric grades *elimination logic*, not conclusions. Each answer needs: what I check first and in what order, what I rule out and on what evidence, the root-cause hypothesis, the fix, and how I validate it before it reaches the client.

- C1 — quantisation accuracy collapse: at least three independent root causes, each with a test that distinguishes it from the others.
- C2 — one camera of twelve offset, worse at edges: what the pattern alone localises, and how to confirm without physical access. Note the direct connection to Snippet 1 — this is the same defect class, and saying so explicitly is worth doing.
- C3 — silent three-month degradation: two or three causes with confirming evidence, plus a monitoring signal with a **specific threshold** that would have fired inside two weeks.

**Part D** — edge and air-gapped deployment.
- D1: 8 cameras × 15 fps = 120 fps aggregate. Show the arithmetic, and separate the *throughput* budget from the *per-frame latency* budget — they are different constraints and conflating them is the common error.
- D2: model family and precision, why the combination fits, and the single first measurement to take.
- D3: air-gap retraining loop — operator flags a detection, feedback physically crosses the gap, new model validated before replacing the running one.
- D4: rollback plan with detection latency and trigger signal.
- D5: the part I am least confident about, and what would resolve it. Answer this genuinely; a candid answer here scores better than false confidence, and the brief says so.

Prefer "I would benchmark X before committing" to a confident unverified number. The brief asks for exactly that.

**Commit separately:** `docs: part C production failure diagnoses` and `docs: part D edge deployment design`

---

## 15. README  `[ ]`

Every `TODO: measure` replaced with a real value. Required sections:

- Hardware and software versions
- Dataset: per-class instance counts, image counts per split, capture methodology, scene-based split rationale
- Annotation: link to the guide, the model-assisted workflow and the correction ratio
- Training config quoted from `resolved_config.json`
- Validation metrics, held-out test metrics, and the gap between them
- ONNX parity: the actual numbers and the method
- The three-way trade-off table
- Exact reproduction commands, in order
- **Assumptions** — every judgement call
- **Known gaps** — everything incomplete and why
- **Confidence statement** — would these numbers hold on a different machine, and why or why not
- Recording link

**Gate.** Someone unfamiliar with the repo can reproduce the results from the README alone, without asking a question.

**Commit:** `docs: complete README with all measured results`

---

## 16. Screen recording  `[ ]`

6–8 minutes. Presentation quality is explicitly not graded; narrating a live pipeline is.

1. Dataset and a sample of annotations. **Point out one genuinely ambiguous label** — the plugged-in brick or the open case is the natural choice, and reference the rule from `ANNOTATION_GUIDE.md` that resolved it.
2. Run ONNX inference on a validation image, live.
3. Run `benchmark.py` and let the numbers appear on screen.
4. Talk through the Part A decision I am least sure about.

Everything shown must match what is in the repo. Record after the final push so the numbers on screen and the numbers in the README are the same.

**Commit:** `docs: add screen recording link`

---

## 17. Final check before submitting  `[ ]`

- [ ] `git log --oneline` shows the real sequence, including at least one failed run and its fix
- [ ] No number in README lacks a producing command
- [ ] Weights and ONNX committed, or linked with a SHA256
- [ ] `download_data.py` fetches the dataset and the checksum verifies from a clean clone
- [ ] ANSWERS.md has all nine sections (B1–B3, C1–C3, D1–D5)
- [ ] Assumptions and Known Gaps are both non-empty — an empty Known Gaps section on a 24-hour assignment is not credible
- [ ] Recording link resolves
- [ ] Repository is public or shared with the sending address

---

## If time runs short

The brief gives the priority order explicitly. Follow it:

1. Dataset + a model that actually trained, with honest validation numbers (stages 2–7)
2. Parts B and C written answers — 40% between them (stage 14, Part B already done)
3. Export and benchmark, even if only FP32 vs FP16 on CPU (stages 8–11)
4. A4 failure analysis and Part D (stages 13–14)

Then write down precisely what was cut, in Known Gaps. The brief states that section is read closely.

---

## Preparation for the live round

Not a deliverable, but the round is where this is won or lost. Before the call, be able to do each of these unaided:

- Add a third class to `data.yaml` and start a training run
- Change an augmentation in the training config and explain the expected effect
- Move the confidence threshold and explain what happens to precision and recall, and why they move in opposite directions
- Walk the commit history and explain what each commit changed
- Defend one Part D decision under pushback — pick the weakest one in advance and prepare the honest concession rather than a defence

Every script in this repo should be readable by me without help. Anything I could not modify under observation is a liability, not an asset.
