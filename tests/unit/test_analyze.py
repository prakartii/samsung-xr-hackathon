"""rewind.analyze.summarize: routing rules end to end on hand-made embeddings.

Every embedding is a sum of orthonormal basis vectors, so each cosine is 0 or 1/sqrt(k)
and every softmax is near one-hot (logit scale 100). Expected values are worked out
in each test's comment.
"""

from dataclasses import replace

import numpy as np
import pytest

from rewind.analyze import Text, key_phrase, summarize
from rewind.config import load_config
from rewind.detect.pairing import VNPrior
from rewind.output import to_document
from rewind.schema import validate
from rewind.vocab import load_vocab
from tests.conftest import DEFAULT_CONFIG

CFG = load_config(DEFAULT_CONFIG)
D = 32
T = 60.0
LABELS = tuple(CFG.label.classes)  # 15 classes: 14 FPVSum + cooking
VERBS, NOUNS = ("take", "put"), ("knife", "plate")
# basis: label i -> e_i (i < 15); verbs -> e_20, e_21; nouns -> e_22, e_23
VERB_DIM, NOUN_DIM = {"take": 20, "put": 21}, {"knife": 22, "plate": 23}


def e(*dims: int) -> np.ndarray:
    v = np.zeros(D)
    v[list(dims)] = 1.0
    return v / np.linalg.norm(v)


TEXT = Text(
    logit_scale=100.0,
    label_keys=LABELS,
    label=np.stack([e(i) for i in range(len(LABELS))]),
    verb_keys=VERBS,
    verbs=np.stack([e(VERB_DIM[v]) for v in VERBS]),
    noun_keys=NOUNS,
    nouns=np.stack([e(NOUN_DIM[n]) for n in NOUNS]),
)
# uniform prior over the 2 x 2 pairs: (0 + eps) / (2 * (0 + eps)) = 0.5
PRIOR = VNPrior.from_counts(VERBS, NOUNS, np.zeros((2, 2)), eps=1.0)


def signals(frame_emb: np.ndarray) -> dict:
    t_dense = np.arange(int(T * 4)) / 4.0  # 4 fps
    t_clip = np.arange(int(T), dtype=float)  # 1 fps
    flat = np.zeros(len(t_dense))
    return {
        "flow": {"t": t_dense, "flow": flat, "global_motion": flat},
        "hands": {"t": t_dense, "hands": flat},
        "clip": {"t": t_clip, "clip_emb": np.tile(frame_emb, (len(t_clip), 1))},
    }


COOKING_TAKE_KNIFE = e(LABELS.index("cooking"), VERB_DIM["take"], NOUN_DIM["knife"])
BIKE = e(LABELS.index("bike"))
BIKE_TAKE_KNIFE = e(LABELS.index("bike"), VERB_DIM["take"], NOUN_DIM["knife"])


def strict_errors(result) -> list[str]:
    errors, _ = validate(to_document(result), load_vocab(), filename_stem="v", strict=True)
    return errors


def test_summarize_cooking_video_routes_epic_with_zero_shot_actions():
    # cos(frame, cooking) = 1/sqrt(3), 0 for the other 14 labels -> P(cooking) ~ 1
    # >= 0.5 with margin ~0.5 >= min_conf 0.1 -> EPIC route, label ("cooking", 1.0).
    # Zero-shot: windows(60, 2, 1) = [0,2] .. [58,60] = 59 windows; top_k = min(3, 2)
    # = 2 verbs and 2 nouns each. Neighbouring windows have tIoU 1/3 < 0.5, so each
    # window is its own anchor: 59 proposals x (2 verbs x 2 nouns) = 236 actions.
    # Best pair: P(take) ~ P(knife) ~ 1 -> output actions carry no prior
    # (actions.prior_alpha 0): (1 * 1) ** 0.5 = 1.
    r = summarize("v", T, signals(COOKING_TAKE_KNIFE), TEXT, PRIOR, CFG)

    assert r.label == ("cooking", 1.0)
    assert len(r.actions) == 236
    _seg, verb, noun, score = r.actions[0]
    assert (verb, noun) == ("take", "knife")
    assert score == pytest.approx(1.0, abs=1e-9)
    assert strict_errors(r) == []


def test_summarize_fpvsum_video_has_no_actions_and_zero_shot_label():
    # Frames are pure "bike": P(cooking) ~ 0, margin 0.5 >= 0.1 -> FPVSum route.
    # Label: softmax over the 14 FPVSum classes = 1 / (1 + 13 e^-100) ~ 1 for bike.
    r = summarize("v", T, signals(BIKE), TEXT, PRIOR, CFG)

    assert r.actions == ()
    assert r.label[0] == "bike"
    assert r.label[1] == pytest.approx(1.0, abs=1e-9)
    assert strict_errors(r) == []


def test_summarize_fpvsum_route_never_runs_the_detector():
    def detect():
        raise AssertionError("ActionFormer must not run on the FPVSum route")

    r = summarize("v", T, signals(BIKE), TEXT, PRIOR, CFG, detect)
    assert r.actions == ()


def test_summarize_uncertain_route_always_emits_actions():
    # min_conf 0.6 > the largest possible margin (0.5), so every video is uncertain.
    # P(cooking) ~ 0 < 0.5 -> FPVSum label "bike" (score ~1 >= other_thr), but the
    # uncertain route still runs the detector: same 59 windows x 4 pairs = 236.
    cfg = replace(CFG, routing=replace(CFG.routing, min_conf=0.6))
    r = summarize("v", T, signals(BIKE_TAKE_KNIFE), TEXT, PRIOR, cfg)

    assert r.label[0] == "bike"
    assert len(r.actions) == 236
    assert strict_errors(r) == []


def test_summarize_uses_actionformer_detections_when_available():
    # One verb and one noun detection on the same span (tIoU 1):
    # output score, no prior (actions.prior_alpha 0) = (0.9 * 0.8) ** 0.5 = 0.72 ** 0.5
    dets = ([((3.0, 7.0), "put", 0.9)], [((3.0, 7.0), "plate", 0.8)])
    r = summarize("v", T, signals(COOKING_TAKE_KNIFE), TEXT, PRIOR, CFG, lambda: dets)

    assert len(r.actions) == 1
    seg, verb, noun, score = r.actions[0]
    assert (seg, verb, noun) == ((3.0, 7.0), "put", "plate")
    assert score == pytest.approx(0.72**0.5, abs=1e-12)


def test_summarize_falls_back_to_zero_shot_when_no_features_are_cached():
    # detect() returns None (no SlowFast entry) -> the zero-shot detector: 236 actions.
    r = summarize("v", T, signals(COOKING_TAKE_KNIFE), TEXT, PRIOR, CFG, lambda: None)
    assert len(r.actions) == 236


def test_summarize_short_video_gives_one_highlight():
    # T = 20 s < SHORT_VIDEO_T (27 s): exactly one segment, 4.05 s in min_seg mode.
    t = 20.0
    sig = signals(BIKE)
    sig = {
        "flow": {k: v[: int(t * 4)] for k, v in sig["flow"].items()},
        "hands": {k: v[: int(t * 4)] for k, v in sig["hands"].items()},
        "clip": {k: v[: int(t)] for k, v in sig["clip"].items()},
    }
    r = summarize("v", t, sig, TEXT, PRIOR, CFG)

    assert len(r.highlights) == 1
    s, end = r.highlights[0]
    assert end - s == pytest.approx(4.05, abs=1e-9)


@pytest.mark.parametrize(
    ("key", "phrase"),
    [("take", "take"), ("turn-on", "turn on"), ("board:chopping", "chopping board")],
)
def test_key_phrase_turns_kit_keys_into_prompt_words(key, phrase):
    assert key_phrase(key) == phrase


def test_summarize_cooking_label_uses_expected_coverage_selector():
    # One ActionFormer action on [30, 34]. With flat signals the importance proxy would
    # take the first windows from t = 0 (budget 0.145 * 60 = 8.7 s), so only the
    # coverage selector puts a highlight over [30, 34].
    dets = ([((30.0, 34.0), "put", 0.9)], [((30.0, 34.0), "plate", 0.8)])
    r = summarize("v", T, signals(COOKING_TAKE_KNIFE), TEXT, PRIOR, CFG, lambda: dets)

    assert r.label[0] == "cooking"
    assert any(s < 34.0 and end > 30.0 for s, end in r.highlights)


def test_summarize_fpvsum_label_excludes_cooking_class():
    # cos(frame, cooking) = cos(frame, bike) = cos(frame, ski) = 1/sqrt(3), so over all 15
    # labels P(cooking) = 1/3: margin |1/3 - 0.5| = 1/6 >= 0.1 -> FPVSum route.
    # Over the 14 FPVSum classes bike and ski tie at 1/2 (1/3 each if cooking leaked in);
    # bike comes first in label order and wins.
    frame = e(LABELS.index("cooking"), LABELS.index("bike"), LABELS.index("ski"))
    r = summarize("v", T, signals(frame), TEXT, PRIOR, CFG)

    assert r.label[0] == "bike"
    assert r.label[1] == pytest.approx(0.5, abs=1e-6)


def test_summarize_actionformer_failure_falls_back_to_zero_shot_actions():
    # An ActionFormer error (bad features, torch mismatch) must cost the detector, not
    # the video: the zero-shot detector still gives the 236 actions.
    def detect():
        raise RuntimeError("feature shape mismatch")

    r = summarize("v", T, signals(COOKING_TAKE_KNIFE), TEXT, PRIOR, CFG, detect)
    assert len(r.actions) == 236


def test_summarize_mismatched_dense_grids_raise():
    # flow and hands come from one pass on the same grid; a mismatch means a broken
    # cache and must fail loudly (the pipeline then writes the fallback).
    sig = signals(BIKE)
    sig["hands"] = {"t": sig["hands"]["t"][:-1], "hands": sig["hands"]["hands"][:-1]}
    with pytest.raises(ValueError, match="grid"):
        summarize("v", T, sig, TEXT, PRIOR, CFG)


def loud_between(lo: float, hi: float):
    """audio(t): quiet (rms 0.01) except loud (rms 0.5) on [lo, hi); flux 0."""

    def audio(t):
        t = np.asarray(t)
        return {"rms": np.where((t >= lo) & (t < hi), 0.5, 0.01), "flux": np.zeros(len(t))}

    return audio


def test_summarize_fpvsum_highlight_follows_loud_audio():
    # All visual signals are flat, so importance = w_audio_rms * z(log rms) only: high
    # on [40, 50), low elsewhere. Without audio the flat importance ties everywhere and
    # the first window wins ([0, ...]). With audio the one highlight (budget 0.145 * 60
    # = 8.7 s) must sit on the loud stretch: its centre inside [40, 50]. (Smoothing,
    # sigma 2 s, blurs the stretch's edges, so the window may start a little before 40.)
    r = summarize("v", T, signals(BIKE), TEXT, PRIOR, CFG, audio=loud_between(40.0, 50.0))

    assert len(r.highlights) == 1
    s, e_ = r.highlights[0]
    assert 40.0 <= (s + e_) / 2 <= 50.0


def test_summarize_epic_route_never_loads_audio():
    def audio(t):
        raise AssertionError("audio must not be decoded on the EPIC route")

    r = summarize("v", T, signals(COOKING_TAKE_KNIFE), TEXT, PRIOR, CFG, audio=audio)
    assert r.label[0] == "cooking"


def test_summarize_audio_failure_keeps_fpvsum_highlights():
    def audio(t):
        raise RuntimeError("no audio stream")

    r = summarize("v", T, signals(BIKE), TEXT, PRIOR, CFG, audio=audio)
    assert len(r.highlights) >= 1
    assert strict_errors(r) == []


# prior from counts take/put x knife/plate = [[0, 8], [0, 8]], eps 1:
# prior(v, knife) = 1 / 10 = 0.1, prior(v, plate) = 9 / 10 = 0.9
PLATE_PRIOR = VNPrior.from_counts(VERBS, NOUNS, np.array([[0.0, 8.0], [0.0, 8.0]]), eps=1.0)
KNIFE_OR_PLATE = (
    [((3.0, 7.0), "take", 0.9)],
    [((3.0, 7.0), "knife", 0.8), ((3.0, 7.0), "plate", 0.3)],
)


def test_summarize_output_actions_scored_without_the_prior_by_default():
    # actions.prior_alpha 0: score = (p_v * p_n) ** 0.5
    #   take knife (0.9 * 0.8) ** 0.5 = 0.72 ** 0.5 = 0.84853; take plate 0.27 ** 0.5 = 0.51962
    r = summarize(
        "v", T, signals(COOKING_TAKE_KNIFE), TEXT, PLATE_PRIOR, CFG, lambda: KNIFE_OR_PLATE
    )

    (_, v0, n0, s0), (_, v1, n1, s1) = r.actions
    assert (v0, n0, v1, n1) == ("take", "knife", "take", "plate")
    assert (s0, s1) == (pytest.approx(0.72**0.5, abs=1e-12), pytest.approx(0.27**0.5, abs=1e-12))


def test_summarize_output_actions_with_prior_alpha_one_follow_the_prior():
    # actions.prior_alpha 1: take knife (0.72 * 0.1) ** 0.5 = 0.26833,
    # take plate (0.27 * 0.9) ** 0.5 = 0.243 ** 0.5 = 0.49295 -> plate first
    cfg = replace(CFG, actions=replace(CFG.actions, prior_alpha=1.0))
    r = summarize(
        "v", T, signals(COOKING_TAKE_KNIFE), TEXT, PLATE_PRIOR, cfg, lambda: KNIFE_OR_PLATE
    )

    (_, _, n0, s0), (_, _, n1, s1) = r.actions
    assert (n0, n1) == ("plate", "knife")
    assert (s0, s1) == (pytest.approx(0.243**0.5, abs=1e-12), pytest.approx(0.072**0.5, abs=1e-12))
