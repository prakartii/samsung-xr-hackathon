"""The numpy NMS port must match ActionFormer's C++ extension exactly.

No compiler here, so the C++ code (csrc/nms_cpu.cpp) is transcribed literally below,
loop for loop in float32, as the reference; the vectorized port is compared to it on
hand cases and many random inputs (ties included).
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_PATH = (
    Path(__file__).resolve().parents[2] / "code/third_party/actionformer/libs/utils/nms_1d_py.py"
)
_spec = importlib.util.spec_from_file_location("nms_1d_py", _PATH)
nms_py = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nms_py)

f32 = np.float32


# --- literal transcription of csrc/nms_cpu.cpp ----------------------------------------


def ref_nms(segs, order, thr):
    x1, x2 = segs[:, 0].astype(f32), segs[:, 1].astype(f32)
    areas = x2 - x1 + f32(1e-6)
    n = len(segs)
    select = [True] * n
    for _i in range(n):
        if not select[_i]:
            continue
        i = order[_i]
        for _j in range(_i + 1, n):
            if not select[_j]:
                continue
            j = order[_j]
            inter = max(f32(0), min(x2[i], x2[j]) - max(x1[i], x1[j]))
            ovr = inter / (areas[i] + areas[j] - inter)
            if ovr >= f32(thr):
                select[_j] = False
    return np.array([order[k] for k in range(n) if select[k]], dtype=np.int64)


def ref_softnms(segs, scores, thr, sigma, min_score, method):
    x1, x2 = list(segs[:, 0].astype(f32)), list(segs[:, 1].astype(f32))
    sc = list(scores.astype(f32))
    areas = [a - b + f32(1e-6) for a, b in zip(x2, x1, strict=True)]
    n = len(sc)
    inds = list(range(n))
    dets = np.zeros((n, 3), dtype=f32)
    i = 0
    while i < n:
        max_score, max_pos = sc[i], i
        for pos in range(i + 1, n):
            if max_score < sc[pos]:
                max_score, max_pos = sc[pos], pos
        ix1, ix2, iscore = x1[max_pos], x2[max_pos], sc[max_pos]
        dets[i] = (ix1, ix2, iscore)
        iarea, iind = areas[max_pos], inds[max_pos]
        x1[max_pos], x2[max_pos], sc[max_pos] = x1[i], x2[i], sc[i]
        areas[max_pos], inds[max_pos] = areas[i], inds[i]
        x1[i], x2[i], sc[i], areas[i], inds[i] = ix1, ix2, iscore, iarea, iind
        pos = i + 1
        while pos < n:
            inter = max(f32(0), min(ix2, x2[pos]) - max(ix1, x1[pos]))
            ovr = inter / (iarea + areas[pos] - inter)
            w = f32(1)
            if method == 0:
                w = f32(0) if ovr >= f32(thr) else f32(1)
            elif method == 1:
                w = f32(1) - ovr if ovr >= f32(thr) else f32(1)
            else:
                w = f32(math.exp(-(ovr * ovr) / f32(sigma)))
            sc[pos] = f32(sc[pos] * w)
            if sc[pos] < f32(min_score):
                x1[pos], x2[pos], sc[pos] = x1[n - 1], x2[n - 1], sc[n - 1]
                areas[pos], inds[pos] = areas[n - 1], inds[n - 1]
                n -= 1
                pos -= 1
            pos += 1
        i += 1
    return np.array(inds[:n], dtype=np.int64), dets[:n]


def _random_case(rng, n, tie_frac=0.3):
    starts = rng.uniform(0, 100, n).astype(f32)
    lens = rng.uniform(0.5, 10, n).astype(f32)
    segs = np.stack([starts, starts + lens], axis=1)
    scores = rng.uniform(0, 1, n).astype(f32)
    ties = rng.random(n) < tie_frac  # force duplicate scores
    scores[ties] = f32(0.5)
    return segs, scores


# --- hand cases -----------------------------------------------------------------------


def test_softnms_gaussian_two_overlapping_segments_hand_computed():
    # A = [0, 10] score 0.9, B = [5, 15] score 0.8. IoU = 5 / (10 + 10 - 5) = 1/3
    # (the 1e-6 length terms are negligible). Gaussian sigma 0.5:
    # B -> 0.8 * exp(-(1/9) / 0.5) = 0.8 * exp(-0.2222) = 0.8 * 0.800737 = 0.640590
    segs = np.array([[0, 10], [5, 15]], dtype=f32)
    inds, dets = nms_py.softnms_np(segs, np.array([0.9, 0.8], f32), 0.1, 0.5, 0.001, 2)
    assert inds.tolist() == [0, 1]
    np.testing.assert_allclose(dets[:, 2], [0.9, 0.640590], atol=1e-5)


def test_softnms_hard_method_drops_overlap_above_threshold():
    # method 0: B's IoU 1/3 >= 0.3 -> weight 0 -> below min_score -> dropped
    segs = np.array([[0, 10], [5, 15]], dtype=f32)
    inds, dets = nms_py.softnms_np(segs, np.array([0.9, 0.8], f32), 0.3, 0.5, 0.001, 0)
    assert inds.tolist() == [0]


def test_nms_keeps_disjoint_drops_overlapping():
    # [0,10] 0.9 keeps; [1,10] 0.8 has IoU 9/10 >= 0.5 -> dropped; [20,30] disjoint keeps
    segs = np.array([[0, 10], [1, 10], [20, 30]], dtype=f32)
    order = np.array([0, 1, 2])
    assert nms_py.nms_np(segs, order, 0.5).tolist() == [0, 2]


def _exact_iou(a, b):
    # The IoU exactly as the C++ computes it in float32, so a threshold can sit on it.
    areas = [f32(s[1]) - f32(s[0]) + f32(1e-6) for s in (a, b)]
    inter = max(f32(0), min(f32(a[1]), f32(b[1])) - max(f32(a[0]), f32(b[0])))
    return inter / (areas[0] + areas[1] - inter)


def test_nms_iou_equal_to_threshold_suppresses():
    # C++: `if (ovr >= iou_threshold) select = false` -> equality suppresses.
    segs = np.array([[0, 10], [5, 15]], dtype=f32)
    thr = float(_exact_iou(segs[0], segs[1]))
    assert nms_py.nms_np(segs, np.array([0, 1]), thr).tolist() == [0]


def test_softnms_hard_iou_equal_to_threshold_drops():
    segs = np.array([[0, 10], [5, 15]], dtype=f32)
    thr = float(_exact_iou(segs[0], segs[1]))
    inds, _ = nms_py.softnms_np(segs, np.array([0.9, 0.8], f32), thr, 0.5, 0.001, 0)
    assert inds.tolist() == [0]


def test_softnms_empty_input():
    inds, dets = nms_py.softnms_np(np.zeros((0, 2), f32), np.zeros(0, f32), 0.1, 0.4, 0.001, 2)
    assert inds.shape == (0,) and dets.shape == (0, 3)


# --- equivalence with the C++ transcription --------------------------------------------


@pytest.mark.parametrize("method", [0, 1, 2])
@pytest.mark.parametrize("seed", range(25))
def test_softnms_matches_cpp_reference(method, seed):
    rng = np.random.default_rng(seed)
    segs, scores = _random_case(rng, int(rng.integers(1, 60)))
    # ActionFormer EPIC: iou 0.1, sigma 0.4, min_score 0.001; a high min_score forces
    # many drops, which exercises the swap-with-last compaction.
    min_score = [0.001, 0.3][seed % 2]
    want_i, want_d = ref_softnms(segs, scores, 0.1, 0.4, min_score, method)
    got_i, got_d = nms_py.softnms_np(segs, scores, 0.1, 0.4, min_score, method)
    np.testing.assert_array_equal(got_i, want_i)
    np.testing.assert_allclose(got_d, want_d, rtol=0, atol=1e-6)


@pytest.mark.parametrize("seed", range(25))
def test_nms_matches_cpp_reference(seed):
    rng = np.random.default_rng(100 + seed)
    segs, scores = _random_case(rng, int(rng.integers(1, 60)))
    order = np.argsort(-scores, kind="stable")
    np.testing.assert_array_equal(nms_py.nms_np(segs, order, 0.3), ref_nms(segs, order, 0.3))
