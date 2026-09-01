# Object Detection: charger_brick / earphone_case

Two-class object detector (YOLO11n) trained on a self-captured dataset,
exported to ONNX, quantised to INT8, and benchmarked — with a scene-level
train/val/test split so near-duplicate frames never leak across splits.

> **Convention in this README:** every number marked `TODO: measure` is filled
> in only by pasting the output of the named script, run on the hardware below.
> Nothing here is estimated.

## Hardware & environment

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM |
| CPU | Intel i5-12450H |
| RAM | 16 GB |
| OS | Windows 11 |
| Python | TODO: measure (`python --version`) |
| torch / CUDA | TODO: measure (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`) |
| ultralytics | TODO: measure (`pip show ultralytics`) |
| onnxruntime | TODO: measure (`pip show onnxruntime`) |

## Dataset

- ~150 images photographed by me, resized to 1280px long edge
  (`scripts/resize_images.py`), two classes: `charger_brick` (0),
  `earphone_case` (1). Labels made in makesense.ai (YOLO txt), rules in
  [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md).
- Filenames encode the capture scene (`scene01_a.jpg`, ...). Frames within a
  scene are near-duplicates, so **the split is by scene, not by image**
  (`scripts/split_dataset.py`), 70/15/15, seed 42. The full assignment is
  committed in `data/split_manifest.json`.
- The test split was evaluated **exactly once**, after all model and
  threshold choices were frozen.
- Annotation flow: ~40 images hand-labelled → seed model → `prelabel.py` on
  the rest → every pre-labelled box manually reviewed. `verify_labels.py`
  passes with 0 hard errors on the final dataset: TODO: paste summary.

### Split & instance counts (paste from `split_dataset.py` / `verify_labels.py`)

| Split | Scenes | Images | charger_brick | earphone_case |
|-------|--------|--------|---------------|---------------|
| train | TODO: measure | TODO: measure | TODO: measure | TODO: measure |
| val | TODO: measure | TODO: measure | TODO: measure | TODO: measure |
| test | TODO: measure | TODO: measure | TODO: measure | TODO: measure |

## Training

Config: YOLO11n, imgsz 640, batch 8, AMP, patience 25 — full resolved config
as actually run is in `runs/detect/<run>/resolved_config.json`. Quote it, not
intentions:

- Epochs run (early stop?): TODO: measure
- Wall-clock training time: TODO: measure (printed by `train.py`)
- Peak VRAM: TODO: measure (from training log)

### Validation metrics (Ultralytics, best.pt)

| Class | P | R | mAP@0.5 | mAP@0.5:0.95 |
|-------|---|---|---------|--------------|
| charger_brick | TODO: measure | TODO: measure | TODO: measure | TODO: measure |
| earphone_case | TODO: measure | TODO: measure | TODO: measure | TODO: measure |
| all | TODO: measure | TODO: measure | TODO: measure | TODO: measure |

## ONNX export & parity

Export: opset 12, batch 1 fixed, simplified (`export_onnx.py`).
I/O tensors: TODO: paste from `export_onnx.py` output.

Parity was verified by feeding the same preprocessed tensor to PyTorch (FP32,
eval) and ONNX Runtime over 10 validation images (`verify_parity.py`):

- Worst raw max abs diff: TODO: measure (threshold 1e-3, script exits non-zero above it)
- Worst raw mean abs diff: TODO: measure
- Worst post-NMS box coordinate diff: TODO: measure px

## Quantisation & the FP32 / INT8 trade-off

INT8 static quantisation calibrated on **train-split images only** (using val
or test images for calibration would leak evaluation data — `quantize.py`
refuses such paths). Both models below evaluated on the val split by the
**same** script, `evaluate_onnx.py`; latency from `benchmark.py`
(CPU, batch 1, 20 warmup + 200 timed iterations).

| Model | Size (MB) | mAP@0.5 | mAP@0.5:0.95 | Mean latency (ms) | Median (ms) | p95 (ms) |
|-------|-----------|---------|--------------|-------------------|-------------|----------|
| FP32 ONNX | TODO: measure | TODO: measure | TODO: measure | TODO: measure | TODO: measure | TODO: measure |
| INT8 ONNX | TODO: measure | TODO: measure | TODO: measure | TODO: measure | TODO: measure | TODO: measure |
| FP16 ONNX (fallback, if used) | TODO: measure | TODO: measure | TODO: measure | TODO: measure | TODO: measure | TODO: measure |

Execution provider / threads: TODO: paste from `benchmark.py` header.

### Held-out test set (run exactly once)

| Model | mAP@0.5 | mAP@0.5:0.95 |
|-------|---------|--------------|
| FP32 ONNX | TODO: measure | TODO: measure |
| INT8 ONNX | TODO: measure | TODO: measure |

## Failure analysis

Honest accounting of where the model breaks (with example images where
possible). See also `notes/hard_images.md`.

- TODO: list observed failure modes after evaluation (e.g. specific scenes,
  occlusion, lighting, confusions between classes, missed small instances).
- TODO: confusion cases: how many val/test images have false positives /
  false negatives, and what do they have in common?

## Reproduction

```bash
# 0. Environment (Windows, Python 3.10+). Install CUDA torch first:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 1. Get the dataset (images; labels + split manifest are in git)
python scripts/download_data.py

# 2. (Only when rebuilding from originals) resize + split + verify
python scripts/resize_images.py --src data/raw --dst data/resized
python scripts/split_dataset.py --images data/resized --labels data/labels_all
python scripts/verify_labels.py --dataset data/dataset

# 3. Train
python scripts/train.py --name train

# 4. Export + parity
python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt
python scripts/verify_parity.py --weights runs/detect/train/weights/best.pt --onnx runs/detect/train/weights/best.onnx --images data/dataset/images/val

# 5. Quantise (train-split calibration only)
python scripts/quantize.py --onnx runs/detect/train/weights/best.onnx --calib-images data/dataset/images/train

# 6. Evaluate both models the same way + benchmark
python scripts/evaluate_onnx.py --onnx runs/detect/train/weights/best.onnx --images data/dataset/images/val --labels data/dataset/labels/val
python scripts/evaluate_onnx.py --onnx runs/detect/train/weights/best_int8.onnx --images data/dataset/images/val --labels data/dataset/labels/val
python scripts/benchmark.py --fp32 runs/detect/train/weights/best.onnx --quant runs/detect/train/weights/best_int8.onnx --image data/dataset/images/val/<any_image>.jpg
```

## Assumptions

- TODO: list assumptions made (e.g. single object scale range, indoor
  lighting only, camera phone quality, no motion blur handling).

## Known gaps

- Small dataset (~150 images, ~N scenes) — metrics have high variance; a
  different split seed would move them. TODO: quantify if time permits.
- TODO: add remaining gaps honestly (things not done, not things done badly).

## Recording

- Walkthrough video: TODO: add link
