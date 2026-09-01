# Environment record

Captured verbatim from the running venv (`artikate/`, Python 3.11.9) on
2026-09-01 by the command in PLAN Phase 1. The README hardware section
quotes this file.

```
torch 2.5.1+cu121 | cuda True | NVIDIA GeForce RTX 3050 Laptop GPU
torchvision 0.20.1+cu121
ultralytics 8.4.137
onnx 1.22.0 | onnxruntime 1.29.0 ['AzureExecutionProvider', 'CPUExecutionProvider']
numpy 2.4.6 | opencv 5.0.0 | pillow 12.3.0
cuda tensor op ok: torch.Size([2, 2])
```

Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), Intel i5-12450H,
16 GB RAM, Windows 11.

Notes:
- onnxruntime is CPU-only (CPUExecutionProvider) — quantisation and latency
  benchmarks run on CPU by design; training runs on the GPU.
- Two environment defects were caught and fixed during setup, both logged in
  `notes/decisions.md`: a stale machine-wide pip config pointing at the
  retired pypi.ngc.nvidia.com index, and `pip install -r requirements.txt`
  silently replacing CUDA torch with a CPU build (caught by re-running the
  GPU gate, fixed by force-reinstalling 2.5.1+cu121 with --no-deps).
- Reproduction order matters: install CUDA torch AFTER the requirements
  file, or re-verify `torch.cuda.is_available()` afterwards.
