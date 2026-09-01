"""The B3 tests, applied to this repo's own IoU implementation.

ANSWERS.md B3 hand-computes IoU cases and insists on translation
invariance. These run against scripts/evaluate_onnx.py's iou_matrix — the
function every mAP number in the README flows through.

Run:  python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_onnx import iou_matrix  # noqa: E402


def one(a, b) -> float:
    return float(iou_matrix(np.array([a], dtype=np.float32),
                            np.array([b], dtype=np.float32))[0, 0])


def test_identical_box_is_one():
    assert abs(one([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-6


def test_disjoint_diagonal_is_zero():
    # The B3 worked example: the broken implementation returns 0.76 here.
    assert one([0, 0, 100, 100], [500, 500, 600, 600]) == 0.0


def test_disjoint_one_axis_is_zero():
    assert one([0, 0, 10, 10], [20, 0, 30, 10]) == 0.0


def test_half_overlap_is_one_third():
    # inter 50, union 150 -> 1/3
    assert abs(one([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-6


def test_translation_invariance():
    # IoU is purely relative: identical configuration translated 1000 px
    # must give an identical result. The xyxy-as-xywh area bug fails this.
    near = one([0, 0, 10, 10], [5, 0, 15, 10])
    far = one([1000, 1000, 1010, 1010], [1005, 1000, 1015, 1010])
    assert abs(near - far) < 1e-6


def test_always_in_unit_interval():
    rng = np.random.default_rng(42)
    for _ in range(500):
        v = rng.integers(0, 2000, size=8)
        a = np.array([[min(v[0], v[2]), min(v[1], v[3]),
                       max(v[0], v[2]), max(v[1], v[3])]], dtype=np.float32)
        b = np.array([[min(v[4], v[6]), min(v[5], v[7]),
                       max(v[4], v[6]), max(v[5], v[7])]], dtype=np.float32)
        val = float(iou_matrix(a, b)[0, 0])
        assert 0.0 <= val <= 1.0 + 1e-6
