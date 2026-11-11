"""Pure part of detect/actionformer.py: model output -> detections (no torch here)."""

import numpy as np
import pytest

from rewind.detect.actionformer import feature_times, to_detections

KEYS = ("take", "put", "wash")


def test_feature_times_hand_computed():
    # (i * 16 + 16) / 30: i = 0 -> 0.5333, i = 1 -> 1.0667, i = 2 -> 1.6
    np.testing.assert_allclose(feature_times(3), [16 / 30, 32 / 30, 48 / 30], atol=1e-12)


def test_to_detections_maps_index_to_key_and_sorts_by_score():
    out = to_detections(
        np.array([[1.0, 2.0], [3.0, 5.0]]), np.array([0.2, 0.9]), np.array([2, 0]), KEYS, 10.0
    )
    assert out == [
        ((3.0, 5.0), "take", pytest.approx(0.9)),
        ((1.0, 2.0), "wash", pytest.approx(0.2)),
    ]


def test_to_detections_clamps_to_duration_and_drops_empty():
    # [8, 12] -> [8, 10]; [11, 12] is entirely past T = 10 -> empty -> dropped
    out = to_detections(
        np.array([[8.0, 12.0], [11.0, 12.0]]), np.array([0.5, 0.5]), np.array([1, 1]), KEYS, 10.0
    )
    assert [d[0] for d in out] == [(8.0, 10.0)]


def test_to_detections_drops_unknown_label_index():
    out = to_detections(np.array([[0.0, 1.0]]), np.array([0.5]), np.array([3]), KEYS, 10.0)
    assert out == []


def test_to_detections_tie_order_is_start_then_key():
    out = to_detections(
        np.array([[5.0, 6.0], [1.0, 2.0], [1.0, 2.0]]),
        np.array([0.5, 0.5, 0.5]),
        np.array([0, 1, 0]),
        KEYS,
        10.0,
    )
    assert [(d[0][0], d[1]) for d in out] == [(1.0, "put"), (1.0, "take"), (5.0, "take")]
