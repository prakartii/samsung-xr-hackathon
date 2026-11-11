"""CLIP math and signal batching with a fake encoder (no torch in the default suite)."""

import dataclasses

import numpy as np
import pytest

from rewind.features.clip import (
    ClipParams,
    class_embeddings,
    class_prompts,
    clip_cache_version,
    clip_signals,
    ensemble,
    l2_normalize,
    zero_shot_probs,
)

PARAMS = ClipParams(model="ViT-B-16", weights="weights/clip.safetensors", width=448, batch_size=32)


# --- l2_normalize -------------------------------------------------------------------


def test_l2_normalize_3_4_gives_0_6_0_8():
    # |(3, 4)| = 5 -> (0.6, 0.8)
    np.testing.assert_allclose(l2_normalize(np.array([[3.0, 4.0]])), [[0.6, 0.8]], atol=1e-12)


def test_l2_normalize_zero_row_stays_zero():
    np.testing.assert_array_equal(l2_normalize(np.zeros((1, 3))), np.zeros((1, 3)))


# --- prompts and ensemble -----------------------------------------------------------


def test_class_prompts_formats_every_template_per_class_in_order():
    out = class_prompts({"ski": "skiing", "cooking": "cooking"}, ["a video of {}.", "{}!"])
    assert out == [["a video of skiing.", "skiing!"], ["a video of cooking.", "cooking!"]]


def test_ensemble_two_orthogonal_prompts_gives_normalized_mean():
    # prompts (2, 0) and (0, 1) -> normalized (1, 0), (0, 1) -> mean (0.5, 0.5)
    # -> normalized (1/sqrt 2, 1/sqrt 2) = (0.707107, 0.707107)
    embs = np.array([[[2.0, 0.0], [0.0, 1.0]]])
    np.testing.assert_allclose(ensemble(embs), [[0.707107, 0.707107]], atol=1e-6)


def test_ensemble_normalizes_each_prompt_before_averaging():
    # Without per-prompt normalization, (10, 0) and (0, 1) would average to
    # (5, 0.5) and point almost entirely along x. With it, the result is diagonal.
    embs = np.array([[[10.0, 0.0], [0.0, 1.0]]])
    np.testing.assert_allclose(ensemble(embs), [[0.707107, 0.707107]], atol=1e-6)


# --- zero_shot_probs ----------------------------------------------------------------


def test_zero_shot_probs_scale_1_hand_computed():
    # cos with class 0 = 1, class 1 = 0; softmax(1, 0) = (e/(e+1), 1/(e+1))
    # = (0.731059, 0.268941)
    p = zero_shot_probs(np.array([[1.0, 0.0]]), np.eye(2), logit_scale=1.0)
    np.testing.assert_allclose(p, [[0.731059, 0.268941]], atol=1e-6)


def test_zero_shot_probs_scale_100_is_near_one_hot_and_stable():
    # softmax(100, 0): exp(-100) ~ 3.7e-44 -> (1, 0) without overflow
    p = zero_shot_probs(np.array([[0.0, 2.0]]), np.eye(2), logit_scale=100.0)
    np.testing.assert_allclose(p, [[0.0, 1.0]], atol=1e-12)


def test_zero_shot_probs_huge_scale_does_not_overflow():
    # exp(1000) overflows float64 to inf (inf/inf = nan); subtracting the row max
    # first gives softmax(0, -1000) = (1, 0).
    p = zero_shot_probs(np.array([[1.0, 0.0]]), np.eye(2), logit_scale=1000.0)
    np.testing.assert_allclose(p, [[1.0, 0.0]], atol=1e-12)


def test_zero_shot_probs_rows_sum_to_one():
    rng = np.random.default_rng(0)
    p = zero_shot_probs(rng.normal(size=(5, 8)), rng.normal(size=(3, 8)), logit_scale=100.0)
    np.testing.assert_allclose(p.sum(axis=1), np.ones(5), atol=1e-12)


# --- class_embeddings with a fake encoder -------------------------------------------


class FakeEncoder:
    """Text: one-hot on a fixed index per known word. Image: mean pixel value as a
    2-D direction, so different frames get different embeddings."""

    logit_scale = 100.0
    _words = {"skiing": 0, "cooking": 1}

    def __init__(self):
        self.text_calls = []

    def encode_texts(self, texts):
        self.text_calls.append(list(texts))
        out = np.zeros((len(texts), 2))
        for i, t in enumerate(texts):
            out[i, self._words[next(w for w in self._words if w in t)]] = 1.0
        return out.astype(np.float32)

    def encode_images(self, rgb_frames):
        v = np.array(
            [[float(f[..., 0].mean()) + 1.0, float(f[..., 2].mean()) + 1.0] for f in rgb_frames]
        )
        return l2_normalize(v).astype(np.float32)


def test_class_embeddings_rows_follow_class_order():
    enc = FakeEncoder()
    emb = class_embeddings(enc, {"cooking": "cooking", "ski": "skiing"}, ["a {}", "the {}"])
    # cooking -> index 1, skiing -> index 0
    np.testing.assert_allclose(emb, [[0.0, 1.0], [1.0, 0.0]], atol=1e-12)


def test_routing_style_p_cooking_from_fake_kitchen_frame():
    # A frame whose embedding equals the "cooking" text direction gets P(cooking) ~ 1.
    enc = FakeEncoder()
    classes = {"ski": "skiing", "cooking": "cooking"}
    cls = class_embeddings(enc, classes, ["{}"])
    p = zero_shot_probs(np.array([[0.0, 1.0]]), cls, enc.logit_scale)
    assert p[0, list(classes).index("cooking")] == pytest.approx(1.0, abs=1e-9)


# --- clip_signals -------------------------------------------------------------------


def _frames(n):
    out = []
    for k in range(n):
        f = np.zeros((4, 4, 3), np.uint8)
        f[..., 0] = 10 * k  # blue in BGR -> channel 2 in RGB
        f[..., 2] = 5  # red in BGR -> channel 0 in RGB
        out.append((float(k), f))
    return out


def test_clip_signals_batch_size_does_not_change_embeddings():
    a = clip_signals(_frames(5), FakeEncoder(), batch_size=1)
    b = clip_signals(_frames(5), FakeEncoder(), batch_size=2)
    np.testing.assert_allclose(a["clip_emb"], b["clip_emb"], atol=1e-7)
    np.testing.assert_allclose(a["t"], [0, 1, 2, 3, 4], atol=1e-12)


def test_clip_signals_converts_bgr_to_rgb():
    # Frame 1: BGR blue = 10, red = 5 -> RGB channel 0 = 5, channel 2 = 10.
    # Fake image embedding = normalize(ch0 + 1, ch2 + 1) = normalize(6, 11)
    out = clip_signals(_frames(2), FakeEncoder(), batch_size=4)
    np.testing.assert_allclose(out["clip_emb"][1], l2_normalize(np.array([6.0, 11.0])), atol=1e-6)


def test_clip_signals_output_is_float32_unit_rows():
    out = clip_signals(_frames(3), FakeEncoder(), batch_size=2)
    assert out["clip_emb"].dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out["clip_emb"], axis=1), 1.0, atol=1e-6)


def test_clip_signals_no_frames_raises():
    with pytest.raises(ValueError):
        clip_signals([], FakeEncoder(), batch_size=2)


# --- cache version ------------------------------------------------------------------


def test_clip_cache_version_ignores_batch_size_tracks_the_rest():
    v = clip_cache_version(PARAMS, 1.0, "abc")
    assert v.startswith("clip-v2-")
    assert v == clip_cache_version(dataclasses.replace(PARAMS, batch_size=8), 1.0, "abc")
    assert v != clip_cache_version(dataclasses.replace(PARAMS, width=224), 1.0, "abc")
    assert v != clip_cache_version(PARAMS, 2.0, "abc")
    assert v != clip_cache_version(PARAMS, 1.0, "def")
