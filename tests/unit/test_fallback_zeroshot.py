"""Zero-shot fallback detector on fake embeddings (no CLIP model)."""

import numpy as np
import pytest

from rewind.detect.fallback_zeroshot import (
    ZeroShotParams,
    detect_zeroshot,
    window_embeddings,
    windows,
)

P = ZeroShotParams(window_sec=2.0, stride_sec=1.0, top_k=1, verb_template="{}", noun_template="{}")


def test_windows_hand_computed():
    # T = 5, window 2, stride 1: starts 0, 1, 2, 3 (3 + 2 = 5 fits; 4 + 2 = 6 doesn't)
    assert windows(5.0, 2.0, 1.0) == [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0), (3.0, 5.0)]


def test_windows_short_video_gets_one_clipped_window():
    assert windows(1.5, 2.0, 1.0) == [(0.0, 1.5)]


def test_windows_end_exactly_at_the_video_end():
    # T = 25, 8 s windows, stride 4: grid starts 0, 4, 8, 12, 16 (last ends at 24), so a
    # final window (17, 25) covers the last second, which would otherwise never be seen.
    w = windows(25.0, 8.0, 4.0)

    assert w[-2:] == [(16.0, 24.0), (17.0, 25.0)]


def test_window_embeddings_average_frames_inside():
    # frames at t = 0, 1, 2 with embeddings e0, e1, e2 (axes). Window [0, 2) holds
    # t = 0 and 1 -> mean (0.5, 0.5, 0) -> normalized (0.7071, 0.7071, 0)
    t = np.array([0.0, 1.0, 2.0])
    out = window_embeddings(t, np.eye(3), [(0.0, 2.0)])
    np.testing.assert_allclose(out, [[0.707107, 0.707107, 0.0]], atol=1e-6)


def test_zeroshot_picks_matching_classes_per_window():
    # 4 s video at 1 fps: frames 0-1 look like verb A + noun X, frames 2-3 like verb B
    # + noun Y (4-d embedding: dims 0/1 = verbs A/B, dims 2/3 = nouns X/Y).
    t = np.arange(4.0)
    emb = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1]], dtype=float)
    verb_text = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    noun_text = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    verbs, nouns = detect_zeroshot(
        t, emb, 4.0, ("A", "B"), verb_text, ("X", "Y"), noun_text, 100.0, P
    )
    by_span_v = {d[0]: d[1] for d in verbs}
    by_span_n = {d[0]: d[1] for d in nouns}
    assert by_span_v[(0.0, 2.0)] == "A" and by_span_v[(2.0, 4.0)] == "B"
    assert by_span_n[(0.0, 2.0)] == "X" and by_span_n[(2.0, 4.0)] == "Y"
    assert all(0.0 <= d[2] <= 1.0 for d in verbs + nouns)
    assert [d[2] for d in verbs] == sorted((d[2] for d in verbs), reverse=True)


def test_zeroshot_top_k_per_window():
    t = np.arange(2.0)
    emb = np.ones((2, 2))
    p3 = ZeroShotParams(2.0, 1.0, top_k=2, verb_template="{}", noun_template="{}")
    verbs, _ = detect_zeroshot(
        t, emb, 2.0, ("A", "B", "C"), np.eye(3, 2), ("X",), np.ones((1, 2)), 1.0, p3
    )
    assert len(verbs) == 2  # one window x top-2


def test_zeroshot_empty_embeddings():
    assert detect_zeroshot(
        np.zeros(0), np.zeros((0, 2)), 3.0, ("A",), np.ones((1, 2)), ("X",), np.ones((1, 2)), 1.0, P
    ) == ([], [])


def test_windows_bad_params_raise():
    with pytest.raises(ValueError):
        windows(5.0, 0.0, 1.0)
