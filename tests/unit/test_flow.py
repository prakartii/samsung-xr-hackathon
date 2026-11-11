"""Unit tests for the flow decomposition and z-scoring (pure part of features/flow.py)."""

import numpy as np
import pytest

from rewind.features.flow import decompose_flow, motion_series, zscore


def _uniform(h, w, dx, dy):
    f = np.zeros((h, w, 2))
    f[..., 0], f[..., 1] = dx, dy
    return f


# --- decompose_flow -----------------------------------------------------------------


def test_decompose_pure_camera_pan_is_all_global():
    # Every pixel moves by (3, 4): median = (3, 4), |median| = 5; residual = 0.
    residual, global_ = decompose_flow(_uniform(10, 10, 3.0, 4.0))
    assert global_ == pytest.approx(5.0, abs=1e-9)
    assert residual == pytest.approx(0.0, abs=1e-9)


def test_decompose_moving_hand_on_static_scene_is_all_residual():
    # 100 px, 10 of them (a "hand") move by (6, 8), |(6,8)| = 10; the rest are still.
    # median = (0, 0) -> global 0; residual = (10 px * 10 + 90 px * 0) / 100 = 1.0
    f = np.zeros((10, 10, 2))
    f[0, :, 0], f[0, :, 1] = 6.0, 8.0
    residual, global_ = decompose_flow(f)
    assert global_ == pytest.approx(0.0, abs=1e-9)
    assert residual == pytest.approx(1.0, abs=1e-9)


def test_decompose_hand_during_pan_separates_both():
    # Pan (3, 4) everywhere; the 10-px hand row moves (3+6, 4+8) = (9, 12).
    # median = (3, 4) -> global 5; residual = 10 * |(6, 8)| / 100 = 1.0
    f = _uniform(10, 10, 3.0, 4.0)
    f[0, :, 0], f[0, :, 1] = 9.0, 12.0
    residual, global_ = decompose_flow(f)
    assert global_ == pytest.approx(5.0, abs=1e-9)
    assert residual == pytest.approx(1.0, abs=1e-9)


def test_decompose_wrong_shape_raises():
    with pytest.raises(ValueError):
        decompose_flow(np.zeros((10, 10)))


# --- zscore -------------------------------------------------------------------------


def test_zscore_hand_computed():
    # mean 2, population std sqrt(((1-2)^2 + 0 + (3-2)^2) / 3) = sqrt(2/3) = 0.816497
    # -> (-1 / 0.816497, 0, 1 / 0.816497) = (-1.224745, 0, 1.224745)
    z = zscore(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(z, [-1.224745, 0.0, 1.224745], atol=1e-6)


def test_zscore_constant_signal_is_zeros():
    np.testing.assert_array_equal(zscore(np.full(5, 7.0)), np.zeros(5))


# --- motion_series ------------------------------------------------------------------


def test_motion_series_assigns_flow_to_later_frame_and_copies_first():
    # 3 frames, 2 flows: flow 0->1 is a pan of 5, flow 1->2 is still.
    # frame 1 = 5, frame 2 = 0, frame 0 copies frame 1 = 5.
    flows = [_uniform(4, 4, 3.0, 4.0), _uniform(4, 4, 0.0, 0.0)]
    residual, global_ = motion_series(flows, n_frames=3)
    np.testing.assert_allclose(global_, [5.0, 5.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(residual, [0.0, 0.0, 0.0], atol=1e-9)


def test_motion_series_single_frame_is_zero():
    residual, global_ = motion_series([], n_frames=1)
    np.testing.assert_array_equal(residual, [0.0])
    np.testing.assert_array_equal(global_, [0.0])


def test_motion_series_wrong_flow_count_raises():
    with pytest.raises(ValueError):
        motion_series([_uniform(4, 4, 1.0, 0.0)], n_frames=3)
