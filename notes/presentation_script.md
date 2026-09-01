# Screen recording script — 6–8 minutes

The brief requires exactly four things: (1) dataset + one genuinely
ambiguous annotation, (2) live ONNX inference on a validation image,
(3) the benchmark running with numbers appearing, (4) the Part A decision
you are least sure about. This script hits them in order, with two of our
strongest stories woven in. Times are targets — total ≈ 7 min.

**Before recording:** open a terminal at the repo root with the venv
active; open `notes/failure_examples/` and `data/dataset/images/val/` in
Explorer; have `README.md` visible in an editor tab. Record AFTER the
final push so on-screen numbers match the README.

---

## 0:00–0:40 — Orientation

> "This is my submission for the Artikate CV assessment. Two classes —
> earphone case and charger brick — 150 photos I took myself: 100 for
> train and validation, and 50 I held out as a test set that was evaluated
> exactly once, at the very end. Everything I'll show is in the repo, and
> every number in the README names the script that produced it. The commit
> history is the real sequence — including three runs that failed and how
> I diagnosed them."

*(Show `git log --oneline | head -30` briefly while saying the last line.)*

## 0:40–2:00 — Dataset + the ambiguous annotation

*(Open `data/dataset/images/val/` in Explorer, scroll. Then open
`notes/failure_examples/err02_IMG_20260901_150211624_MP.jpg`.)*

> "Labels were made in makesense.ai — 208 boxes, and the YOLO, VOC and CSV
> exports agree box-for-box, verified by a script. The genuinely ambiguous
> one is this: I own two chargers and two earphone cases, and one of each
> is white. This white earphone case, photographed top-down on glass, is
> nearly the same shape and colour as the white charger. My annotation
> guide's answer was to label strictly by physical identity, not
> appearance — and the model's worst failures are exactly on this object,
> which I'll show at the end."

> "One thing I caught before training: the scaffolded config had the class
> IDs flipped relative to what makesense actually exported. I found it by
> cross-referencing the YOLO txt against the CSV coordinate-by-coordinate —
> all 208 boxes — not by testing, because flipped IDs train cleanly and
> produce plausible-looking metrics. That's commit 7324505."

## 2:00–2:50 — The split (our strongest discipline story, 30 seconds)

*(Show `notes/leakage_report.md`.)*

> "Frames shot seconds apart are near-duplicates, so I split by scene, not
> by image. Filenames carried no scene IDs and timestamps couldn't separate
> an 87-image continuous burst, so scene identity comes from visual
> clustering — and I verified the property I actually care about with a
> leakage audit: every val and test image compared against every train
> image under two different similarity metrics. Closest cross-split pair:
> Hamming 10 of 64. The audit is the guarantee; the clustering is just the
> means."

## 2:50–4:00 — Live ONNX inference (required beat #2)

*(Type and run, live:)*

```bash
python scripts/detect_folder.py --onnx models/best.onnx --source data/dataset/images/val --out runs/demo_recording
```

*(Open two or three results in `runs/demo_recording/`, including one clean
detection.)*

> "This is the exported ONNX model — opset 12, fixed batch 1 — through ONNX
> Runtime, using the same preprocessing module every script in this repo
> shares, so the parity check, the benchmark and the evaluation can never
> disagree on pixels. Parity against PyTorch: worst raw difference
> 8.85e-4, boxes agreeing within a ten-thousandth of a pixel, measured
> over ten validation images."

## 4:00–5:10 — Benchmark, numbers appearing live (required beat #3)

*(Type and run, live — takes ~30s:)*

```bash
python scripts/benchmark.py --fp32 models/best.onnx --quant models/best_int8.onnx --image data/dataset/images/val/IMG_20260901_145029586_HDR.jpg --warmup 20 --iters 200
```

*(While it runs:)*

> "Twenty warmup iterations discarded, two hundred timed, mean and p95.
> And the honest headline here is that INT8 is slower than FP32 on this
> CPU — about 2.2 times slower. The size win is real, seventy percent
> smaller, and the accuracy cost is 0.033 mAP. But quantising only the
> convolutions leaves dequantize round-trips around each one, and on x86
> that overhead beats the int8 saving. On a Jetson with native int8 paths
> I'd expect this to invert — but I measured what I have, and I'd rather
> show you a true negative than a guessed positive."

> "Getting INT8 working at all was the best bug of the project: the first
> quantised model scored exactly zero mAP. Box coordinates survived,
> every class score was exactly zero — because the YOLO head concatenates
> boxes ranging to 640 with scores up to 1 in one tensor, and one
> per-tensor scale rounds every score away. That's the same failure class
> as my C1 answer, reproduced on my own model, diagnosed in the history."

## 5:10–6:10 — Results and the failure analysis

*(Show the README headline table, then
`notes/failure_examples/err01_IMG_20260901_145627625_HDR.jpg`.)*

> "Validation mAP@0.5 is 0.99 — and I lead the README with mAP@0.5:0.95
> instead, 0.72, because with boxes this large IoU-0.5 is too easy to
> clear to discriminate anything. The held-out test, run once: 0.66. That
> val-to-test gap of about six points is the most informative number in
> the submission — it's the measured optimism of a 21-image validation
> split. And all three worst validation images are the white earphone
> case: confused with the white charger, or missed when they're adjacent.
> The scene split correctly kept those scenes out of training — that's the
> honest cost of a leak-free split at this scale."

## 6:10–7:00 — Least-sure decision (required beat #4) and close

> "The Part A decision I'm least sure about is the scene-clustering
> threshold. I chose 0.10 from the knee of a scan — it's committed, the
> whole scan is in the repo — but at 0.15 I'd have 16 clusters and at
> 0.05 I'd have 84, and the split would differ. My mitigation is that the
> split's guarantee doesn't come from the clustering at all: it comes from
> the leakage audit, which passes under two metrics regardless. If I had
> another day, the first thing I'd do isn't tuning — it's a second capture
> session in different lighting, because the variation audit measured a
> brightness spread of only 22 grey levels, and that, not the model, is
> this dataset's real ceiling."

> "Everything shown is in the repo: 50 commits, the failed runs included.
> Thanks."

---

## Don'ts

- Don't re-run training or anything touching the test set on camera.
- Don't say "about" for a number that's in the README — say the number.
- If something errors live, debug it on camera calmly; that is worth more
  than a clean take.
