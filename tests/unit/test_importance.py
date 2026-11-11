"""FPVSum importance proxy and segment selection (PROGRESS.md P6 tests)."""

import numpy as np
import pytest

from rewind.select.importance import (
    ImportanceParams,
    clip_change,
    gaussian_smooth,
    importance_signal,
    select_segments,
    shots,
)

P = ImportanceParams(
    w_flow=1.0, w_hands=0.0, w_global=0.0, w_scene=0.0, sigma=0.0,
    shot_thr=0.5, min_len=4.05, max_len=6.05, len_step=1.0, stride=0.5,
)  # fmt: skip


# --- clip_change / shots ------------------------------------------------------------


def test_clip_change_hand_computed():
    # e0, e0, e1 (orthogonal): changes at j=1: 1 - 1 = 0, j=2: 1 - 0 = 1; j=0 copies j=1
    e = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(clip_change(e), [0.0, 0.0, 1.0], atol=1e-12)


def test_shots_cut_where_change_exceeds_threshold():
    # 1 fps, cut at t = 3 (embedding flips): shots [0, 3), [3, 6]
    t = np.arange(6.0)
    e = np.array([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
    assert shots(t, e, 6.0, thr=0.5) == [(0.0, 3.0), (3.0, 6.0)]


def test_shots_no_cut_is_one_shot():
    assert shots(np.arange(4.0), np.ones((4, 2)), 4.0, thr=0.5) == [(0.0, 4.0)]


# --- smoothing / signal -------------------------------------------------------------


def test_gaussian_smooth_keeps_constant_and_spreads_spike_symmetrically():
    np.testing.assert_allclose(gaussian_smooth(np.full(20, 3.0), 2.0), 3.0, atol=1e-12)
    x = np.zeros(21)
    x[10] = 1.0
    y = gaussian_smooth(x, 2.0)
    assert y.argmax() == 10
    np.testing.assert_allclose(y[10 - 3], y[10 + 3], atol=1e-12)
    assert y.sum() == pytest.approx(1.0, abs=1e-9)
    # Kernel reaches 3 sigma = 6 samples: weights exp(-i^2 / 8) for i = -6..6 sum to
    # 1 + 2 * (0.882497 + 0.606531 + 0.324652 + 0.135335 + 0.043937 + 0.011109)
    # = 5.008122, so the spike keeps 1 / 5.008122 = 0.199676 at its center.
    assert y[10] == pytest.approx(0.199676, abs=1e-6)


def test_importance_signal_is_weighted_sum_before_smoothing():
    # sigma 0, scene weight 0: imp = 1.0 * flow + 0.5 * hands + 2.0 * global
    p = ImportanceParams(1.0, 0.5, 2.0, 0.0, 0.0, 0.5, 4.05, 6.05, 1.0, 0.5)
    t = np.arange(4) / 4
    imp = importance_signal(
        t, np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0]), np.array([0, 0, 1.0, 0]),
        np.arange(1.0), np.ones((1, 2)), p,
    )  # fmt: skip
    np.testing.assert_allclose(imp, [1.0, 0.5, 2.0, 0.0], atol=1e-12)


# --- select_segments ----------------------------------------------------------------


def _grid(duration, rate=4.0):
    return np.arange(int(duration * rate)) / rate


def test_select_picks_window_around_importance_bump():
    # 60 s video, importance 1 on [30, 34.05), 0 elsewhere; budget 4.05 s -> one window.
    # The only 4.05 s window with mean 1.0 is [30.0, 34.05].
    t = _grid(60.0)
    imp = ((t >= 30.0) & (t < 34.05)).astype(float)
    segs = select_segments(t, imp, [(0.0, 60.0)], 60.0, budget_sec=4.05, p=P)
    assert segs == [(30.0, 34.05)]


def test_select_respects_budget_and_no_overlap():
    t = _grid(100.0)
    imp = np.sin(t / 3.0)
    segs = select_segments(t, imp, [(0.0, 100.0)], 100.0, budget_sec=14.5, p=P)
    assert sum(e - s for s, e in segs) <= 14.5 + 1e-9
    assert all(e - s >= 4.05 - 1e-9 for s, e in segs)
    assert all(a[1] <= b[0] for a, b in zip(segs, segs[1:], strict=False))


def test_select_windows_stay_inside_their_shot():
    # shots [0, 10) and [10, 20); a bump straddling the cut can't be taken across it
    t = _grid(20.0)
    imp = ((t >= 8.0) & (t < 12.0)).astype(float)
    segs = select_segments(t, imp, [(0.0, 10.0), (10.0, 20.0)], 20.0, budget_sec=4.05, p=P)
    ((s, e),) = segs
    assert e <= 10.0 or s >= 10.0


def test_select_short_budget_returns_single_best_window():
    # budget 3 s < min_len: one min_len window (post-processing applies short-video rules)
    t = _grid(20.0)
    imp = (t >= 15.0).astype(float)
    segs = select_segments(t, imp, [(0.0, 20.0)], 20.0, budget_sec=3.0, p=P)
    assert len(segs) == 1 and segs[0][1] - segs[0][0] == pytest.approx(4.05)
    assert segs[0][0] >= 15.0 - 1e-9


def test_params_reject_min_len_below_kit_minimum():
    with pytest.raises(ValueError):
        ImportanceParams(1, 1, 1, 1, 1, 0.5, 4.0, 6.0, 1.0, 0.5)
