"""FPVSum importance proxy and segment selection (PROGRESS.md P6 tests)."""

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from rewind.select.importance import (
    ImportanceParams,
    clip_change,
    fpvsum_segments,
    gaussian_smooth,
    importance_signal,
    select_segments,
    shots,
    window_lengths,
)

P = ImportanceParams(
    w_flow=1.0, w_hands=0.0, w_global_motion=0.0, w_scene_change=0.0, sigma=0.0,
    shot_thr=0.5, min_window=4.05, max_window=6.05, len_step=1.0, stride=0.5,
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


def test_importance_signal_scene_change_term_is_z_scored():
    # Only the scene weight: CLIP e0, e0, e1 at t = 0, 1, 2 → change [0, 0, 1]
    # (mean 1/3, population std √2/3) → z = [−1/√2, −1/√2, √2].
    p = ImportanceParams(0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 4.05, 6.05, 1.0, 0.5)
    t = np.arange(3.0)
    e = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    zeros = np.zeros(3)

    imp = importance_signal(t, zeros, zeros, zeros, t, e, p)

    np.testing.assert_allclose(imp, [-1 / np.sqrt(2), -1 / np.sqrt(2), np.sqrt(2)], atol=1e-12)


def test_shots_change_equal_to_threshold_is_not_a_cut():
    # A cut needs change > thr: exactly at thr stays one shot, just above it cuts.
    e = np.array([[1.0, 0.0], [1.0, 0.0], [0.6, 0.8]])
    thr = float(clip_change(e)[2])  # 1 - 0.6 = 0.4 (as computed)

    assert shots(np.arange(3.0), e, 3.0, thr=thr) == [(0.0, 3.0)]
    assert shots(np.arange(3.0), e, 3.0, thr=thr - 1e-9) == [(0.0, 2.0), (2.0, 3.0)]


def test_window_lengths_include_max_window_off_the_grid():
    # 4.05 + k · 1.0 → 4.05 … 9.05, then max_window 10.0 itself.
    p = ImportanceParams(1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 4.05, 10.0, 1.0, 0.5)

    assert window_lengths(p) == pytest.approx([4.05, 5.05, 6.05, 7.05, 8.05, 9.05, 10.0], abs=1e-9)


def test_window_lengths_on_the_grid_are_not_duplicated():
    # 4.05, 5.05, 6.05: max_window is already on the grid.
    assert window_lengths(P) == pytest.approx([4.05, 5.05, 6.05], abs=1e-9)


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
    assert all(a[1] <= b[0] for a, b in pairwise(segs))


def test_select_windows_stay_inside_their_shot():
    # shots [0, 10) and [10, 20); a bump straddling the cut can't be taken across it
    t = _grid(20.0)
    imp = ((t >= 8.0) & (t < 12.0)).astype(float)
    segs = select_segments(t, imp, [(0.0, 10.0), (10.0, 20.0)], 20.0, budget_sec=4.05, p=P)
    ((s, e),) = segs
    assert e <= 10.0 or s >= 10.0


def test_select_short_budget_returns_single_best_window():
    # budget 3 s < min_window: one min_window window (post-processing applies short-video rules)
    t = _grid(20.0)
    imp = (t >= 15.0).astype(float)
    segs = select_segments(t, imp, [(0.0, 20.0)], 20.0, budget_sec=3.0, p=P)
    assert len(segs) == 1 and segs[0][1] - segs[0][0] == pytest.approx(4.05)
    assert segs[0][0] >= 15.0 - 1e-9


def test_params_reject_min_window_below_kit_minimum():
    with pytest.raises(ValueError):
        ImportanceParams(1, 1, 1, 1, 1, 0.5, 4.0, 6.0, 1.0, 0.5)


def test_fpvsum_segments_follow_motion_burst_within_budget():
    # 100 s video, 4 fps. Residual flow bursts on [60, 66); everything else flat; one
    # shot (constant CLIP). Budget 0.145 * 100 = 14.5 s -> segments cover the burst.
    t = _grid(100.0)
    burst = ((t >= 60.0) & (t < 66.0)).astype(float)
    flow = {"t": t, "flow": burst, "global_motion": np.zeros_like(t)}
    hands = {"t": t, "hands": np.zeros_like(t)}
    clip = {"t": np.arange(100.0), "clip_emb": np.ones((100, 4))}
    p = ImportanceParams(1.0, 0.0, 0.0, 0.0, 1.0, 0.15, 4.05, 10.0, 1.0, 0.5)
    segs = fpvsum_segments(flow, hands, clip, 100.0, 0.145, p)
    assert sum(e - s for s, e in segs) <= 14.5 + 1e-9
    assert any(s <= 61.0 and e >= 65.0 for s, e in segs)


def test_fpvsum_segments_reject_mismatched_dense_grids():
    t = _grid(10.0)
    flow = {"t": t, "flow": np.zeros_like(t), "global_motion": np.zeros_like(t)}
    hands = {"t": t[:-1], "hands": np.zeros(len(t) - 1)}
    clip = {"t": np.arange(10.0), "clip_emb": np.ones((10, 4))}
    with pytest.raises(ValueError):
        fpvsum_segments(flow, hands, clip, 10.0, 0.145, P)


def test_importance_signal_audio_terms_are_z_scored_log_rms_and_flux():
    # Only audio, sigma 0. rms [e^-1, e^-1, e^1] - floor-free enough: log ~ [-1, -1, 1]
    # (floor 1e-4 shifts it by < 3e-4) -> z = [-1/√2, -1/√2, √2]; flux [0, 0, 3] -> same
    # z. imp = 1.0 * z_rms + 0.5 * z_flux = 1.5 * [-1/√2, -1/√2, √2].
    p = replace(
        ImportanceParams(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 4.05, 6.05, 1.0, 0.5),
        w_audio_rms=1.0, w_audio_flux=0.5,
    )  # fmt: skip
    t = np.arange(3.0)
    zeros = np.zeros(3)
    audio = {"rms": np.exp([-1.0, -1.0, 1.0]), "flux": np.array([0.0, 0.0, 3.0])}

    imp = importance_signal(t, zeros, zeros, zeros, t, np.ones((3, 2)), p, audio)

    z = np.array([-1 / np.sqrt(2), -1 / np.sqrt(2), np.sqrt(2)])
    np.testing.assert_allclose(imp, 1.5 * z, atol=1e-3)


def test_importance_signal_without_audio_ignores_audio_weights():
    # Same inputs with and without audio=None: audio weights have no effect without audio.
    p = replace(
        ImportanceParams(1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 4.05, 6.05, 1.0, 0.5),
        w_audio_rms=1.0, w_audio_flux=1.0,
    )  # fmt: skip
    t = np.arange(4) / 4
    flow = np.array([1.0, 0, 0, 0])
    zeros = np.zeros(4)
    imp = importance_signal(t, flow, zeros, zeros, np.arange(1.0), np.ones((1, 2)), p, None)
    np.testing.assert_allclose(imp, flow, atol=1e-12)
