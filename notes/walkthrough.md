# Script walkthrough — live-round prep

One line per script: what it does, and the one thing about it you will be
asked. Keep current as scripts change.

| Script | What it does | The question you'll get |
|---|---|---|
| `common.py` | The single letterbox/preprocess/decode/NMS pipeline every ONNX-path script imports | "Why one shared pipeline?" — so FP32/INT8/PyTorch comparisons can never diverge in preprocessing; Snippet 1's bug class is impossible by construction |
| `resize_images.py` | Long-edge 1280 resize, EXIF baked in, never overwrites originals | "Why is resizing after labelling safe?" — YOLO labels are normalised; only the absolute-pixel CSV/XML go stale |
| `check_exif_orientation.py` | Per-image: EXIF-applied dims vs the CSV's labelling dims | "What breaks if this fails?" — labels and pixels in different orientations; every box lands rotated 90° |
| `cluster_scenes.py` | pHash + single-linkage union-find scene clustering, threshold from an auditable scan, timestamps only as a weak prior | "Why not split by filename/timestamp?" — no scene IDs exist and an 87-image continuous burst has no time gaps |
| `split_dataset.py` | Scene-aware split; greedy-balanced mode assigns whole clusters to test/val by image deficit | "Why split by scene?" — near-duplicate frames across splits measure memorisation, not generalisation |
| `check_leakage.py` | dHash every val/test image against every train image; flags near-duplicate cross-split pairs | "Why is this the real gate?" — it verifies the property the split is supposed to guarantee, independently of how the split was made |
| `audit_variation.py` | Measures brightness spread, background diversity, box-area span | "Why measure?" — the 14-minute single-session concern becomes a number, not a guess |
| `verify_labels.py` | Orphans both ways, coordinate ranges, degenerate boxes, class IDs, area histogram | "What did it catch?" — nothing; 0 errors on 208 boxes, which is itself evidence |
| `render_labels.py` | Draws GT boxes for eyeball review | "Cheapest check that exists" — a mirrored/mislabelled box is unmissable rendered |
| `train.py` | YOLO11n, 4GB-VRAM defaults, dumps resolved_config.json, prints wall-clock | "What actually ran?" — quote resolved_config.json, not the CLI intent |
| `prelabel.py` | Model-assisted labelling with save_txt | "Was it used?" — no; all 150 hand-labelled; kept for future data |
| `export_onnx.py` | Opset 12, fixed batch 1, simplify, prints I/O tensors, copies deliverables to models/ | "Why fixed batch 1?" — deployment is single-stream; dynamic shapes complicate quantisation and TensorRT parity |
| `verify_parity.py` | Same tensor through torch (fp32, eval) and ORT; raw max/mean abs diff + post-NMS px diff over 10 val images | "Why compare raw tensors AND boxes?" — raw diff proves numerics; box diff proves nothing downstream amplifies it |
| `quantize.py` | ORT static INT8, QDQ, per-channel, MinMax; calibration ONLY from train; FP16 fallback | "Why refuse val/test calibration?" — activation ranges tuned on eval images leak eval data into the model |
| `benchmark.py` | 20 warmup + 200 timed iters, batch 1, mean/median/p95/std, logs provider+threads | "Why warmup?" — first iterations pay allocation/JIT costs that aren't steady-state |
| `evaluate_onnx.py` | Self-contained P/R/mAP@0.5/mAP@0.5:0.95 with greedy IoU matching | "Why hand-rolled?" — FP32 and INT8 must be scored by identical code; cross-checked against Ultralytics in Phase 6 |
| `render_predictions.py` | Predictions vs GT overlays, errNN_ filename prefix sorts worst first | "How were hard images picked?" — post-hoc by measured error, stated plainly |
| `download_data.py` | Drive fetch + pinned SHA256, refuses on mismatch | "Why the hash?" — reproducers must provably evaluate the same images |
| `build_scene_map.py` | (superseded by cluster_scenes.py) timestamp-gap draft clustering | "Why superseded?" — couldn't split the 87-image continuous burst |
