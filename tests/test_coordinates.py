"""The B1.4 test, applied to this repo's own pipeline.

ANSWERS.md B1 calls the letterbox round-trip 'the single highest-value test
in a detection codebase'. This is that test, running against
scripts/common.py — non-square shapes, boxes at the corners where scale
drift is maximal, odd dimensions forcing asymmetric padding.

Run:  python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import boxes_to_original, letterbox  # noqa: E402


@pytest.mark.parametrize("h,w", [
    (640, 640),    # square — the case that hides letterbox bugs
    (1080, 1920),  # landscape
    (1920, 1080),  # portrait
    (480, 1280),   # extreme wide
    (721, 1279),   # odd dimensions, forces asymmetric pad
    (1280, 960),   # this dataset's actual resized shape
])
def test_letterbox_roundtrip(h, w):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _, gain, pad = letterbox(img, 640)

    # GT boxes at the full frame, both corners, and the centre.
    gt = np.array([
        [0, 0, w - 1, h - 1],
        [0, 0, 20, 20],
        [w - 21, h - 21, w - 1, h - 1],
        [w // 2 - 10, h // 2 - 10, w // 2 + 10, h // 2 + 10],
    ], dtype=np.float32)

    # Forward-map into canvas space exactly as the resize does.
    canvas = gt.copy()
    canvas[:, [0, 2]] = gt[:, [0, 2]] * gain + pad[0]
    canvas[:, [1, 3]] = gt[:, [1, 3]] * gain + pad[1]

    recovered = boxes_to_original(canvas, gain, pad, (h, w))
    assert np.max(np.abs(recovered - gt)) < 1.0, "coordinate round-trip drifted > 1 px"


def test_letterbox_output_shape_and_gain():
    img = np.zeros((1280, 960, 3), dtype=np.uint8)
    padded, gain, pad = letterbox(img, 640)
    assert padded.shape == (640, 640, 3)
    assert gain == pytest.approx(0.5)
    assert pad[0] == pytest.approx((640 - 480) / 2)
    assert pad[1] == pytest.approx(0.0)
