"""Hand-presence signal with a fake scorer (no MediaPipe in the default suite)."""

import dataclasses

import numpy as np
import pytest

from rewind.features.hands import HandParams, hands_cache_version, hands_signals, presence

PARAMS = HandParams(
    model="weights/hand_landmarker.task",
    width=320,
    num_hands=2,
    min_detection_conf=0.3,
    min_presence_conf=0.3,
    min_tracking_conf=0.3,
)


# --- presence -----------------------------------------------------------------------


def test_presence_no_hand_is_zero():
    assert presence([]) == pytest.approx(0.0, abs=1e-12)


def test_presence_two_hands_takes_max():
    # max(0.6, 0.9) = 0.9
    assert presence([0.6, 0.9]) == pytest.approx(0.9, abs=1e-12)


def test_presence_out_of_range_score_is_clipped():
    assert presence([1.2]) == pytest.approx(1.0, abs=1e-12)


# --- hands_signals ------------------------------------------------------------------


def _frames(n, fps=4.0):
    # BGR frames whose blue channel encodes the index, so the fake can see which frame
    # it got (and that it got RGB: blue lands in channel 2 after conversion).
    out = []
    for k in range(n):
        f = np.zeros((4, 4, 3), np.uint8)
        f[..., 0] = k  # blue in BGR
        out.append((k / fps, f))
    return out


def test_hands_signals_maps_each_frame_through_scorer():
    # Fake: frame k has hands with scores [k / 10]; k = 0 has no hand.
    def fake(rgb, t):
        k = int(rgb[0, 0, 2])  # blue is channel 2 in RGB
        return [k / 10] if k else []

    out = hands_signals(_frames(4), fake)
    np.testing.assert_allclose(out["t"], [0.0, 0.25, 0.5, 0.75], atol=1e-12)
    np.testing.assert_allclose(out["hands"], [0.0, 0.1, 0.2, 0.3], atol=1e-12)


def test_hands_signals_passes_timestamps_to_scorer_in_order():
    seen = []

    def fake(rgb, t):
        seen.append(t)
        return []

    hands_signals(_frames(3, fps=2.0), fake)
    assert seen == pytest.approx([0.0, 0.5, 1.0], abs=1e-12)


def test_hands_signals_no_frames_raises():
    with pytest.raises(ValueError):
        hands_signals([], lambda rgb, t: [])


def test_hands_cache_version_tracks_params_rate_and_weights():
    v = hands_cache_version(PARAMS, 4.0, "abc")
    assert v == hands_cache_version(PARAMS, 4.0, "abc")
    assert v.startswith("hands-v2-")
    assert v != hands_cache_version(dataclasses.replace(PARAMS, num_hands=1), 4.0, "abc")
    assert v != hands_cache_version(PARAMS, 2.0, "abc")
    assert v != hands_cache_version(PARAMS, 4.0, "def")
