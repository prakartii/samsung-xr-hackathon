"""Farneback edge of features/flow.py on small synthetic frames built in code.

Farneback's exact output can't be hand-computed, so these tests assert behaviour a
caller relies on: a camera pan raises global motion, a moving object raises residual
motion, and timestamps pass through untouched.
"""

import dataclasses

import cv2
import numpy as np
import pytest

from rewind.features.flow import (
    FlowParams,
    flow_cache_version,
    flow_signals,
    raw_motion,
    zscore,
)

PARAMS = FlowParams(
    width=80, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2
)
H, W = 60, 80


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 0)  # trackable texture


def _bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _timed(frames, fps=4.0):
    return [(k / fps, f) for k, f in enumerate(frames)]


def test_flow_signals_camera_pan_raises_global_motion():
    # Frames 0-4 identical (still camera), frames 5-9 shift the whole texture 2 px
    # per frame (camera pan). Global motion must be higher during the pan.
    tex = _texture()
    frames = [_bgr(tex)] * 5 + [_bgr(np.roll(tex, 2 * k, axis=1)) for k in range(1, 6)]
    out = flow_signals(_timed(frames), PARAMS)
    still, pan = out["global_motion"][1:5], out["global_motion"][6:]
    assert pan.min() > still.max()


def test_flow_signals_moving_patch_raises_residual_not_global():
    # Static background; a 20x20 textured patch moves 2 px per frame in frames 5-9
    # (a "hand"). Residual motion must be higher while it moves.
    bg, patch = _texture(0), _texture(1)[:20, :20]
    frames = []
    for k in range(10):
        f = bg.copy()
        x = 10 + 2 * max(0, k - 4)
        f[20:40, x : x + 20] = patch
        frames.append(_bgr(f))
    out = flow_signals(_timed(frames), PARAMS)
    still, moving = out["flow"][1:5], out["flow"][6:]
    assert moving.min() > still.max()


def test_raw_motion_pan_magnitude_is_shift_and_residual_near_zero():
    # Whole texture shifts 2 px per frame: every flow vector ~ (2, 0), so
    # global = |median| ~ 2.0 px and residual = mean |v - median| ~ 0.
    # Tolerance 0.3 px absorbs Farneback's smoothing at the wrapped image border.
    tex = _texture()
    frames = [_bgr(np.roll(tex, 2 * k, axis=1)) for k in range(5)]
    _, residual, global_ = raw_motion(_timed(frames), PARAMS)
    np.testing.assert_allclose(global_[1:], 2.0, atol=0.3)
    assert residual[1:].max() < 0.3


def test_raw_motion_patch_moves_residual_but_not_global():
    # 20x20 patch moving 2 px/frame on a still 80x60 background: the median vector
    # is (0, 0) -> global ~ 0; residual ~ (400 px * 2 px) / 4800 px = 0.167 px.
    # Farneback spreads motion past the patch edges, so allow [0.08, 0.5].
    bg, patch = _texture(0), _texture(1)[:20, :20]
    frames = []
    for k in range(5):
        f = bg.copy()
        f[20:40, 10 + 2 * k : 30 + 2 * k] = patch
        frames.append(_bgr(f))
    _, residual, global_ = raw_motion(_timed(frames), PARAMS)
    assert global_[1:].max() < 0.05
    assert residual[1:].min() > 0.08
    assert residual[1:].max() < 0.5


def test_flow_signals_are_zscores_of_matching_raw_series():
    # Pan for 3 frames, then the patch moves: the two raw series differ in shape, so
    # each output key must be the z-score of *its* raw series, not the other one.
    tex, patch = _texture(0), _texture(1)[:20, :20]
    frames = [_bgr(np.roll(tex, 2 * k, axis=1)) for k in range(3)]
    for k in range(3):
        f = np.roll(tex, 4, axis=1).copy()
        f[20:40, 10 + 2 * k : 30 + 2 * k] = patch
        frames.append(_bgr(f))
    _, residual, global_ = raw_motion(_timed(frames), PARAMS)
    out = flow_signals(_timed(frames), PARAMS)
    np.testing.assert_allclose(out["flow"], zscore(residual), atol=1e-9)
    np.testing.assert_allclose(out["global_motion"], zscore(global_), atol=1e-9)


def test_flow_signals_returns_input_timestamps_and_equal_lengths():
    frames = [_bgr(_texture())] * 6
    out = flow_signals(_timed(frames, fps=2.0), PARAMS)
    np.testing.assert_allclose(out["t"], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5], atol=1e-12)
    assert len(out["flow"]) == len(out["global_motion"]) == 6


def test_flow_signals_no_frames_raises():
    with pytest.raises(ValueError):
        flow_signals([], PARAMS)


def test_flow_cache_version_is_stable_and_tracks_params():
    v = flow_cache_version(PARAMS, dense_fps=4.0)
    assert v == flow_cache_version(PARAMS, dense_fps=4.0)
    assert v.startswith("flow-v2-")
    assert v != flow_cache_version(dataclasses.replace(PARAMS, winsize=21), dense_fps=4.0)
    assert v != flow_cache_version(PARAMS, dense_fps=2.0)


def test_flow_online_resizes_wider_frames_to_param_width():
    # The single decoding pass hands flow 448-px frames; FlowOnline must downsize to
    # p.width itself, giving the same signal as frames decoded at p.width directly.
    tex = _texture()
    small = [_bgr(np.roll(tex, 2 * k, axis=1)) for k in range(5)]
    wide = [cv2.resize(f, (2 * W, 2 * H), interpolation=cv2.INTER_NEAREST) for f in small]
    _, _, g_small = raw_motion(_timed(small), PARAMS)
    _, _, g_wide = raw_motion(_timed(wide), PARAMS)
    # nearest-upscaled then area-downscaled frames equal the originals exactly
    np.testing.assert_allclose(g_wide, g_small, atol=1e-6)
