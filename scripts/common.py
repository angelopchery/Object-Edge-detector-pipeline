"""Shared preprocessing/postprocessing used by verify_parity.py, quantize.py,
benchmark.py and evaluate_onnx.py.

One implementation, imported everywhere, so FP32/INT8/PyTorch numbers are
always produced through the exact same pixel pipeline. This mirrors the
Ultralytics LetterBox transform used at inference (center-padded, pad
value 114, no upscaling restriction disabled: scaleup=True).
"""

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def letterbox(img_bgr: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize + center-pad to (size, size). Returns (image, gain, (pad_w, pad_h))."""
    h, w = img_bgr.shape[:2]
    gain = min(size / h, size / w)
    new_w, new_h = round(w * gain), round(h * gain)
    pad_w, pad_h = (size - new_w) / 2, (size - new_h) / 2

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, gain, (pad_w, pad_h)


def preprocess(image_path: Path, size: int = 640) -> tuple[np.ndarray, float, tuple[float, float], tuple[int, int]]:
    """Image file -> NCHW float32 [0,1] RGB tensor. Returns (tensor, gain, pad, (orig_h, orig_w))."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    orig_h, orig_w = img_bgr.shape[:2]
    padded, gain, pad = letterbox(img_bgr, size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor), gain, pad, (orig_h, orig_w)


def boxes_to_original(boxes_xyxy: np.ndarray, gain: float, pad: tuple[float, float],
                      orig_shape: tuple[int, int]) -> np.ndarray:
    """Map xyxy boxes from letterboxed 640-space back to original-image pixels."""
    out = boxes_xyxy.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / gain
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / gain
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, orig_shape[1])
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, orig_shape[0])
    return out


def decode_and_nms(raw_output: np.ndarray, conf_thres: float = 0.25,
                   iou_thres: float = 0.45) -> np.ndarray:
    """Decode a YOLO11 detection head output and run class-aware NMS.

    Input: (1, 4+nc, N) — xywh in letterbox pixels + per-class scores.
    Returns (M, 6): x1, y1, x2, y2, score, class — still in letterbox space.
    """
    pred = raw_output[0].T  # (N, 4+nc)
    boxes_xywh, scores_all = pred[:, :4], pred[:, 4:]
    class_ids = scores_all.argmax(axis=1)
    scores = scores_all.max(axis=1)

    keep = scores >= conf_thres
    boxes_xywh, scores, class_ids = boxes_xywh[keep], scores[keep], class_ids[keep]
    if len(boxes_xywh) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    xyxy = np.empty_like(boxes_xywh)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # Class-aware NMS via the classic coordinate-offset trick.
    offset = class_ids.astype(np.float32)[:, None] * 4096.0
    idxs = cv2.dnn.NMSBoxes(
        bboxes=np.hstack([xyxy[:, :2] + offset, xyxy[:, 2:] - xyxy[:, :2]]).tolist(),
        scores=scores.tolist(), score_threshold=conf_thres, nms_threshold=iou_thres,
    )
    idxs = np.array(idxs).reshape(-1)
    dets = np.hstack([xyxy[idxs], scores[idxs, None], class_ids[idxs, None].astype(np.float32)])
    # Sort by descending score for deterministic comparisons.
    return dets[dets[:, 4].argsort()[::-1]].astype(np.float32)
