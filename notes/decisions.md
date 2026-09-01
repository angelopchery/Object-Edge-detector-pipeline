# Decisions log

Append-only record of non-obvious choices: what was decided, alternatives,
why. Raw material for the README's Assumptions section and the live round.

## 2026-09-01 — Environment

- **Replaced the Python 3.14 venv with 3.11.9** rather than trying 3.14:
  CUDA torch and onnxruntime wheel coverage for 3.14 is uncertain; 3.11 is
  the safest widely-supported floor. Alternative: 3.12 (kept as fallback if
  any wheel is missing).
- **Machine has a stale pip config** (`C:\ProgramData\pip\pip.ini` and the
  user-level pip.ini) adding the retired `pypi.ngc.nvidia.com` extra index,
  which no longer resolves and made every install retry against dead DNS.
  Fixed with a venv-level `artikate/pip.ini` override (index-url pypi.org,
  empty extra-index-url) instead of editing machine config — least invasive,
  reproducible from the README.

## 2026-09-01 — Scene identity

- **Scene identity comes from visual clustering (pHash + single-linkage),
  not filenames or timestamps.** Filenames are camera originals with no
  scene encoding; timestamp clustering leaves an 87-image continuous burst
  that cannot be split by time. Hand-partitioning 87 near-continuous frames
  is not reliable either. The clustering threshold is chosen from an
  auditable scan (notes/scene_clustering.md) and the actual guarantee is the
  cross-split leakage audit (notes/leakage_report.md), not the clustering.
- **split_dataset accepts a scene map marked `validated_by: leakage-audit`**
  (from cluster_scenes.py) as an alternative to human `reviewed: true` —
  the audit is a stronger, reproducible guarantee than eyeballing.

## 2026-09-01 — Split shape

- **Greedy-balanced split (--test-frac 0.33, --val-frac 0.2)**: whole scenes
  assigned largest-first to whichever of test/val has the biggest image
  deficit, rest to train. Alternative was pure seeded shuffling of scene IDs,
  which on ~20-40 unevenly sized clusters can produce badly skewed split
  sizes. Class balance is checked after the fact and reported, never fixed
  by resampling (PLAN Phase 3 rule).

## 2026-09-01 — Scene clustering method (measured)

- **pHash rejected by measurement**: temporally adjacent frames (same
  arrangement, <=5s apart) averaged Hamming 29.4/64 vs 31.5 for frames
  >60s apart — no separation, because handheld close-up reframing changes
  global image structure between consecutive shots. Auto-threshold on the
  pHash scan produced a degenerate 113-image chained cluster.
- **HSV colour-histogram correlation adopted**: adjacent median distance
  0.236 vs distant 0.916 — clean separation. Auto knee chose 0.10 → 40
  clusters (sizes 59/25/16/10/3/2/2 + 33 singletons).
- The large clusters span the whole session: they are recurring
  **backgrounds**, not time blocks — consistent with the variation audit's
  10 background groups. Splitting on background-level visual context is the
  conservative choice for leakage.
- **Split greedy amended**: a cluster is assigned to test/val only if it
  fits the remaining deficit (+10% slack), else train — prevents the
  59-image cluster swallowing the test split and starving training.

## 2026-09-01 — Environment (continued)

- **pip clobbered CUDA torch**: `pip install -r requirements.txt` replaced
  torch 2.5.1+cu121 with 2.13.0+cpu (CUDA False). Caught by re-running the
  GPU gate after the install, per the Phase 1 decision table; fixed with
  `pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url
  https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps`.
  Lesson recorded: on this stack, verify `torch.cuda.is_available()` after
  ANY pip operation.

## 2026-09-01 — Phase 0 batch answered

- **Class semantics confirmed**: 0=earphone_case (green in makesense),
  1=charger_brick (red). Correction to the inferred guide: FOUR physical
  objects exist — two chargers (one black, one white) and two earphone
  cases — never more than one unit of a class per frame. Intra-class
  appearance variation is real (black vs white charger); generalisation
  beyond these four units is still unmeasured.
- **No supplementary capture**: proceeding with the 150 images. The narrow
  single-lighting-regime finding (brightness std 21.8, range 83-170)
  becomes the lead Known Gaps item, quantified.
- **Guide rules confirmed**: boxes exclude cable and shadow; no unlabelled
  background instances anywhere; the case never appears open.

## 2026-09-01 — Phase 4 plausibility gate

- Training: 87 epochs (early stop, best at 62), 11m 14s wall-clock, peak
  ~1.4 GB VRAM. Best-val: P 0.899, R 0.968, mAP@0.5 0.988, mAP@0.5:0.95
  0.755 (earphone_case 0.769, charger_brick 0.742).
- mAP@0.5 exceeded the 0.97 tripwire → tighter leakage re-audit (threshold
  12) flagged 3 pairs; visual inspection showed same-location,
  different-object frames — allowed by a scene split, not leakage. Verdict:
  metric is real, dataset is easy; README leads with mAP@0.5:0.95 and says
  why mAP@0.5 is not the discriminator.

## 2026-09-01 — Phase 6 evaluation cross-check (val split)

- Ultralytics (best.pt, rect val): mAP@0.5 0.988, mAP@0.5:0.95 0.755
- evaluate_onnx.py (best.onnx, square 640 letterbox): 0.956 / 0.722
- Gap 0.032 — inside the 0.01-0.05 "convention difference" band. Tested
  NMS IoU 0.45 vs Ultralytics' 0.7: negligible (0.955/0.725), so the
  drivers are (a) rect-batched val preprocessing vs fixed square 640
  letterbox and (b) 101-point vs all-point AP interpolation, amplified by
  the 21-image val set where one missed detection costs ~0.05 class-AP.
- Verdict: proceed. FP32-vs-INT8 comparability is what matters and both
  sides use evaluate_onnx.py; both toolchains' numbers are reported in the
  README with this explanation.

## 2026-09-01 — Phase 7 quantisation: a real C1-style collapse, diagnosed

- First INT8 attempt produced an INVALID GRAPH: per-channel DequantizeLinear
  needs the `axis` attribute (opset >= 13) but the export is pinned to
  opset 12. Fix: upgrade a temp copy to opset 13 for quantisation only.
- Second attempt loaded but scored mAP 0.000. Diagnosis by raw-output
  comparison: box coords survived (7.6..635.7, matching FP32) but every
  class score was exactly 0. Mechanism: the YOLO head CONCATENATES boxes
  (range 0..640) and sigmoid scores (0..1) into one output tensor; a single
  per-tensor scale for that tensor (~2.5) quantises every score to zero.
  This is precisely ANSWERS.md C1 Cause 3, reproduced on my own model.
- Fix: op_types_to_quantize=["Conv"] — quantise the compute-heavy convs,
  leave mixed-range head arithmetic in float.
- Measured result (evaluate_onnx.py, val, same code as FP32):
  FP32 0.956/0.722 -> INT8 0.933/0.689 (drop 0.023 / 0.033), size
  10.11 MB -> 3.00 MB (-70.3%). FP16 fallback not needed (drop << 0.15).

## 2026-09-01 — Phase 7 benchmark: INT8 measured SLOWER on this CPU

- CPU (i5-12450H, CPUExecutionProvider, ORT default threads, batch 1,
  20 warmup + 200 iters, on mains, no other load):
  FP32 mean 29.80 ms (p95 31.53) vs INT8 mean 64.40 ms (p95 70.47) —
  0.46x "speedup", i.e. 2.2x slower.
- Explanation: Conv-only QDQ inserts Quantize/Dequantize pairs around
  every conv; on this x86 CPU the Q/DQ overhead exceeds the int8 compute
  saving. The size win (-70%) is real; the latency win is not, on this
  hardware. Reported as measured — connects directly to ANSWERS.md D5:
  ORT-on-x86 numbers bound nothing about the deployment accelerator.

## 2026-09-01 — Phase 8 held-out test (run exactly once, no tuning after)

- FP32: mAP@0.5 0.849, mAP@0.5:0.95 0.659 (earphone_case 0.779/0.618,
  charger_brick 0.920/0.700)
- INT8: mAP@0.5 0.793, mAP@0.5:0.95 0.609
- Val -> test gap (FP32): 0.956 -> 0.849 mAP@0.5 (-0.107) and
  0.722 -> 0.659 (-0.063). The val figures were optimistic by roughly
  this much; earphone_case degrades most. Reported verbatim.
