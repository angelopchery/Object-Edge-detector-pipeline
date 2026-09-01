# Screen recording script — 6–8 minutes

**Style:** a story with proof, not a tech readout. Every claim is followed
by something visible on screen. Speak like you're walking a colleague
through your work — plain sentences, real numbers, no rushing. The brief
requires four beats (dataset + one ambiguous label, live ONNX inference,
live benchmark, least-sure decision); they're all here, inside the story.

**Screen prep before recording:**
- Terminal at repo root, venv active, font size up
- Explorer windows ready: `data/dataset/images/val/`, `notes/failure_examples/`
- Editor tabs open: `README.md`, `notes/leakage_report.md`, `notes/scene_clustering.md`
- Do a silent dry run of both live commands once so nothing surprises you

---

## 0:00–0:45 — The hook: what this is, and the one rule I followed

> "Hi, I'm Angelo. Over the last day I built an object detector from
> nothing — I photographed the objects myself, labelled every image by
> hand, trained the model, compressed it, and stress-tested it. Two
> everyday objects: an earphone case and a charger brick.
>
> I followed one rule the whole way: **never report a number I can't
> prove.** Every figure I'll say out loud is produced by a script in this
> repository, and the git history shows the real journey — including
> three things that broke, and how I tracked each one down. I'd actually
> argue the failures are the best part, and I'll show you why."

*(While talking, slowly scroll `git log --oneline` — 50 commits.)*

## 0:45–2:00 — The dataset, and the label that made me stop and think

*(Open `data/dataset/images/val/` and scroll the photos.)*

> "150 photos, taken on my phone in one session. Here's the thing most
> people miss with a small dataset like this: photos taken seconds apart
> are basically twins. If one twin lands in training and the other in
> validation, the model isn't being tested — it's being shown the answers.
> The score looks great and means nothing.
>
> So I never split by photo. I grouped photos into *scenes* — same place,
> same arrangement — and each scene went entirely to one side. And because
> a promise isn't proof, I wrote an auditor that compares every validation
> and test image against every training image, two different ways, and
> reports the closest matches."

*(Flip to `notes/leakage_report.md`, point at the closest-pair lines.)*

> "This is the audit. The most similar cross-split pair differs by 10 bits
> out of 64 — comfortably not a duplicate. That file is my proof the split
> is honest.
>
> Now the label that genuinely made me stop."

*(Open `notes/failure_examples/err02_IMG_20260901_150211624_MP.jpg`.)*

> "I own two of each object — and one charger and one case are both white.
> Top-down, on glass, this white earphone case looks remarkably like the
> white charger. I labelled by what the object *is*, not what it looks
> like — and kept the rule written down so all 208 boxes follow it. Hold
> that thought about these two white objects, because it comes back at
> the end."

## 2:00–2:45 — The catch that saved the whole project

> "Before training, one check paid for the entire process. My config said
> class zero is the charger. The labelling tool's export said the
> opposite. Nothing would ever have crashed — the model would have trained
> happily, learned everything perfectly, and swapped the two names in
> every single prediction. Plausible numbers, silently wrong product.
>
> I caught it by cross-referencing the three export formats the labelling
> tool produces — matching all 208 boxes coordinate by coordinate. That's
> the kind of defect I go hunting for *before* it exists, because after it
> ships, nobody thinks to look. It's commit 7324505."

## 2:45–4:00 — Watch it work, live *(required beat: live ONNX inference)*

> "Enough claims — let's run it. This is the exported ONNX model, the same
> artifact you'd actually deploy, not the training framework."

*(Type and run, live:)*

```bash
python scripts/detect_folder.py --onnx models/best.onnx --source data/dataset/images/val --out runs/demo_recording
```

*(Open two results — one clean case, one clean charger.)*

> "Green is the earphone case, blue is the charger, and the number is how
> sure the model is. Before trusting this exported model, I verified it
> gives the *same answers* as the original — same image into both, and the
> outputs agree to within one ten-thousandth of a pixel on final box
> positions. So nothing was lost in translation from training to
> deployment — measured, not assumed."

## 4:00–5:15 — The compression story *(required beat: benchmark, live)*

> "For edge deployment you want the model small. I compressed it to
> a quarter of the precision — and my first attempt scored exactly zero.
> Not low. Zero.
>
> Here's the diagnosis, and I love how clean it is: the model outputs box
> *positions*, numbers up to 640, and *confidences*, numbers up to 1, glued
> into one block. Compression picks one step size for the whole block —
> sized for 640, every confidence rounds down to nothing. The boxes were
> perfect; the model just lost its voice. So I compressed only the heavy
> compute layers and left that mixed block alone. Accuracy came back
> within three points of the original, at 70% smaller. Let's benchmark
> both, live:"

*(Type and run:)*

```bash
python scripts/benchmark.py --fp32 models/best.onnx --quant models/best_int8.onnx --image data/dataset/images/val/IMG_20260901_145029586_HDR.jpg --warmup 20 --iters 200
```

*(Let the table print. Point at it.)*

> "And here's a result I'm choosing to show you *because* it's
> inconvenient: the compressed model is actually **slower** on my laptop's
> CPU — about two times. The compression bookkeeping costs more than the
> smaller math saves on this particular chip. On a proper edge accelerator
> that flips. I could have hidden that line; instead it's in my README,
> explained — because you'd find out in production anyway, and I'd rather
> be the one who tells you."

## 5:15–6:30 — The failures, with the evidence open

*(Open README headline table.)*

> "Numbers: on validation the model scores 0.99 on the lenient metric —
> and I *lead with the strict one instead*: 0.72, dropping to 0.66 on a
> 50-image test set I locked away and touched exactly once. That gap is
> the most honest number in the project — it's the measured amount by
> which validation flattered me.
>
> And remember the two white objects?"

*(Open `notes/failure_examples/err01_IMG_20260901_145627625_HDR.jpg`.)*

> "Every single one of the model's worst mistakes is the white case:
> called a charger here, missed entirely there when it sits right next to
> the white charger. This isn't a mystery — my split *correctly* kept
> those scenes out of training, so the model barely met that appearance.
> That's the honest price of a clean split on 150 images, and I know
> exactly what fixes it: twenty minutes of new photos of those two
> objects together. Not a smarter model — better data. Knowing which of
> those two levers to pull is most of this job."

## 6:30–7:15 — The decision I'd defend the softest, and the close

> "The decision I'm least certain about: how aggressively I grouped photos
> into scenes. I picked the threshold from a measured scan — it's all
> committed — but a different threshold gives a different grouping. My
> protection is that the split's guarantee never came from the grouping;
> it comes from that leakage audit, which passes regardless.
>
> Everything you've seen is in the repository: 50 commits in real order,
> the failures and their fixes included, tests covering the exact bug
> classes from Part B, and a README where every number names the script
> that produced it. Thanks for watching — I'm looking forward to digging
> into it live."

---

## Delivery notes

- **Pace:** the script reads in ~6:45 at a calm pace. Pause after each
  number; let the tables sit on screen for two beats before moving on.
- **Say the numbers plainly**: "zero point seven two", "eight point
  eight five e minus four" → say "less than a thousandth".
- If a live command errors, keep recording and fix it calmly — that
  footage is worth more than a clean take.
- Never touch anything related to the test set on camera.
- The confidence comes from one place: everything you say has a file
  behind it that's already open. You're not claiming — you're pointing.
