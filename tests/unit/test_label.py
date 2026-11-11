"""Label selection with fake CLIP embeddings (PROGRESS.md P6 tests)."""

import numpy as np
import pytest

from rewind.label import choose_label, fpvsum_label, keyframe_indices
from rewind.routing import RouteDecision

KEYS = ("ski", "surfing")
CLASS_EMB = np.eye(2)


def test_keyframe_indices_evenly_spaced():
    # 9 frames, 3 keyframes -> linspace(0, 8, 3) = 0, 4, 8
    assert keyframe_indices(9, 3).tolist() == [0, 4, 8]
    assert keyframe_indices(2, 8).tolist() == [0, 1]


def test_fpvsum_label_picks_argmax_with_mean_probability():
    # 2 keyframes, both exactly "ski" (axis 0). logit scale 1: softmax(1, 0) per frame
    # = (e / (e + 1), 1 / (e + 1)) = (0.731059, 0.268941) -> mean is the same.
    name, score = fpvsum_label(np.array([[1.0, 0.0], [1.0, 0.0]]), CLASS_EMB, KEYS, 1.0, 8)
    assert name == "ski"
    assert score == pytest.approx(0.731059, abs=1e-6)


def test_fpvsum_label_averages_over_keyframes():
    # 3 frames: ski, ski, surfing at scale 100 -> per-frame ~one-hot -> mean (2/3, 1/3)
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    name, score = fpvsum_label(emb, CLASS_EMB, KEYS, 100.0, 8)
    assert name == "ski" and score == pytest.approx(2 / 3, abs=1e-6)


def _route(dataset, p):
    return RouteDecision(dataset=dataset, route_conf=abs(p - 0.5), p_cooking=p)


def test_choose_label_epic_is_cooking_one():
    assert choose_label(_route("epic_kitchens_100", 0.9), 0.5, ("ski", 0.8), 0.3) == (
        "cooking",
        1.0,
    )


def test_choose_label_fpvsum_uses_zero_shot():
    assert choose_label(_route("fpvsum", 0.1), 0.5, ("ski", 0.8), 0.3) == ("ski", 0.8)


def test_choose_label_uncertain_leaning_cooking():
    assert choose_label(_route("unknown", 0.55), 0.5, ("ski", 0.8), 0.3) == ("cooking", 0.55)


def test_choose_label_uncertain_low_confidence_is_other():
    # leaning FPVSum (0.45 < 0.5) and zero-shot score 0.2 < other_thr 0.3 -> other
    assert choose_label(_route("unknown", 0.45), 0.5, ("ski", 0.2), 0.3) == ("other", 0.2)
    # confident enough zero-shot keeps its label
    assert choose_label(_route("unknown", 0.45), 0.5, ("ski", 0.4), 0.3) == ("ski", 0.4)


def test_choose_label_uncertain_exactly_at_threshold_is_cooking():
    # Same tie rule as routing: P(cooking) >= thr -> cooking (a kitchen video keeps
    # its label and actions).
    assert choose_label(_route("unknown", 0.5), 0.5, ("ski", 0.8), 0.3) == ("cooking", 0.5)


def test_choose_label_zero_shot_score_exactly_at_other_thr_keeps_its_label():
    # "other" only below other_thr: a score equal to it (0.3) keeps the zero-shot label.
    assert choose_label(_route("unknown", 0.45), 0.5, ("ski", 0.3), 0.3) == ("ski", 0.3)
