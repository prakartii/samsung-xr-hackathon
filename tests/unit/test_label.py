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


def test_keyframe_indices_zero_means_every_frame():
    # n = 0 -> all 5 frames, in order
    assert keyframe_indices(5, 0).tolist() == [0, 1, 2, 3, 4]


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


def test_fpvsum_label_logprob_aggregate_is_a_normalized_geometric_mean():
    # 2 frames at scale 1: ski (0.731059, 0.268941), surfing (0.268941, 0.731059).
    # Geometric mean per class: sqrt(.731059 * .268941) both -> renormalized (0.5, 0.5);
    # argmax ties -> first key.
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    name, score = fpvsum_label(emb, CLASS_EMB, KEYS, 1.0, 8, aggregate="logprob")
    assert name == "ski" and score == pytest.approx(0.5, abs=1e-6)


def test_fpvsum_label_logprob_aggregate_differs_from_prob_mean():
    # scale 100. Frames 1-2 lean ski: (1, 0.9)/1.3454 = (0.7433, 0.6690), logit gap 7.43
    # -> p(ski) 0.9994, log p(surf) -7.43. Frame 3 is surfing (0, 1): log p(ski) = -100.
    # prob mean: ski (0.9994 + 0.9994 + 0) / 3 = 0.666 -> ski.
    # logprob sums: ski -0.0006 * 2 - 100 = -100.0, surf -7.43 * 2 + 0 = -14.9 -> surfing.
    emb = np.array([[1.0, 0.9], [1.0, 0.9], [0.0, 1.0]])
    assert fpvsum_label(emb, CLASS_EMB, KEYS, 100.0, 8)[0] == "ski"
    assert fpvsum_label(emb, CLASS_EMB, KEYS, 100.0, 8, aggregate="logprob")[0] == "surfing"


def test_fpvsum_label_embedding_aggregate_softmaxes_the_mean_embedding():
    # frames (1, 0) and (0.6, 0.8) (both unit) -> mean (0.8, 0.4) -> unit (0.894, 0.447);
    # scale 1 -> softmax(0.894, 0.447) = 1 / (1 + e^-0.447) = 0.610 for ski.
    emb = np.array([[1.0, 0.0], [0.6, 0.8]])
    name, score = fpvsum_label(emb, CLASS_EMB, KEYS, 1.0, 8, aggregate="embedding")
    assert name == "ski" and score == pytest.approx(
        1 / (1 + np.exp(-(0.8 - 0.4) / np.sqrt(0.8))), abs=1e-4
    )


def test_fpvsum_label_unknown_aggregate_is_an_error():
    with pytest.raises(ValueError):
        fpvsum_label(np.eye(2), CLASS_EMB, KEYS, 1.0, 8, aggregate="median")
