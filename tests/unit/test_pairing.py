"""Verb-noun pairing (PROGRESS.md P4 tests; ARCHITECTURE.md §5.1)."""

import math

import numpy as np
import pytest

from rewind.detect.pairing import (
    PairingParams,
    VNPrior,
    pair_proposals,
    proposals_to_actions,
    tiou_matrix,
)

VERBS = ("take", "cut")
NOUNS = ("knife", "onion")
VOCAB_V, VOCAB_N = set(VERBS), set(NOUNS)
# counts: take-knife 3, take-onion 1, cut-knife 0, cut-onion 4
COUNTS = np.array([[3.0, 1.0], [0.0, 4.0]])
PRIOR = VNPrior.from_counts(VERBS, NOUNS, COUNTS, eps=1.0)
# Plain product: alpha 1, power 1, so scores are easy to hand-compute.
P1 = PairingParams(match_tiou=0.5, top_k=5, alpha=1.0, eps=1.0, score_power=1.0)


# --- prior --------------------------------------------------------------------------


def test_prior_from_counts_hand_computed():
    # take row: (3+1, 1+1) / 6 = (0.6667, 0.3333); cut row: (0+1, 4+1) / 6 = (0.1667, 0.8333)
    np.testing.assert_allclose(PRIOR.matrix, [[4 / 6, 2 / 6], [1 / 6, 5 / 6]], atol=1e-12)
    assert PRIOR.get("cut", "knife") == pytest.approx(1 / 6, abs=1e-12)


def test_prior_zero_eps_rejected():
    with pytest.raises(ValueError):
        VNPrior.from_counts(VERBS, NOUNS, COUNTS, eps=0.0)


# --- tiou ---------------------------------------------------------------------------


def test_tiou_matrix_hand_computed():
    # [0,10] vs [5,15]: inter 5, union 15 -> 1/3; vs [20,30]: 0
    m = tiou_matrix(np.array([[0.0, 10.0]]), np.array([[5.0, 15.0], [20.0, 30.0]]))
    np.testing.assert_allclose(m, [[1 / 3, 0.0]], atol=1e-12)


# --- pair_proposals -----------------------------------------------------------------


def test_pairing_scores_are_product_with_prior():
    # verb take 0.8 on [0,10]; nouns knife 0.5 on [1,10] (tIoU 0.9), onion 0.4 on [0,9] (0.9)
    # take-knife = 0.8 * 0.5 * 4/6 = 0.26667; take-onion = 0.8 * 0.4 * 2/6 = 0.10667
    props = pair_proposals(
        [((0.0, 10.0), "take", 0.8)],
        [((1.0, 10.0), "knife", 0.5), ((0.0, 9.0), "onion", 0.4)],
        PRIOR, VOCAB_V, VOCAB_N, P1,
    )  # fmt: skip
    assert len(props) == 1
    pr = props[0]
    assert pr.segment == (0.0, 10.0)
    assert [(v, n) for v, n, _ in pr.pairs] == [("take", "knife"), ("take", "onion")]
    np.testing.assert_allclose([s for *_, s in pr.pairs], [0.266667, 0.106667], atol=1e-6)


def test_pairing_prior_changes_ranking():
    # Same noun probabilities (0.5 each). For "cut", prior favours onion (5/6) over
    # knife (1/6): cut-onion = 0.9 * 0.5 * 5/6 = 0.375 ranks above cut-knife = 0.075.
    props = pair_proposals(
        [((0.0, 10.0), "cut", 0.9)],
        [((0.0, 10.0), "knife", 0.5), ((0.0, 10.0), "onion", 0.5)],
        PRIOR, VOCAB_V, VOCAB_N, P1,
    )  # fmt: skip
    assert [(v, n) for v, n, _ in props[0].pairs] == [("cut", "onion"), ("cut", "knife")]
    np.testing.assert_allclose([s for *_, s in props[0].pairs], [0.375, 0.075], atol=1e-9)


def test_pairing_noun_below_match_tiou_is_not_paired():
    # noun [6,16] vs verb [0,10]: inter 4, union 16 -> 0.25 < 0.5 -> no proposal
    props = pair_proposals(
        [((0.0, 10.0), "take", 0.8)], [((6.0, 16.0), "knife", 0.9)], PRIOR, VOCAB_V, VOCAB_N, P1
    )
    assert props == []


def test_pairing_out_of_vocabulary_pairs_dropped():
    prior = VNPrior.from_counts(("take", "zap"), NOUNS, np.ones((2, 2)), eps=1.0)
    props = pair_proposals(
        [((0.0, 10.0), "zap", 0.9), ((0.0, 10.0), "take", 0.5)],
        [((0.0, 10.0), "knife", 0.5)],
        prior, VOCAB_V, VOCAB_N, P1,
    )  # fmt: skip
    # "zap" leads the group but isn't in the kit vocabulary: only take-knife survives
    assert [(v, n) for pr in props for v, n, _ in pr.pairs] == [("take", "knife")]


def test_pairing_out_of_vocabulary_noun_dropped():
    prior = VNPrior.from_counts(VERBS, ("knife", "zork"), np.ones((2, 2)), eps=1.0)
    props = pair_proposals(
        [((0.0, 10.0), "take", 0.8)],
        [((0.0, 10.0), "zork", 0.9), ((0.0, 10.0), "knife", 0.5)],
        prior, VOCAB_V, VOCAB_N, P1,
    )  # fmt: skip
    assert [(v, n) for pr in props for v, n, _ in pr.pairs] == [("take", "knife")]


def test_pairing_same_span_verbs_give_one_proposal():
    # take 0.8 and cut 0.6 on the same span: one proposal anchored on take, with both verbs
    props = pair_proposals(
        [((0.0, 10.0), "take", 0.8), ((0.0, 10.0), "cut", 0.6)],
        [((0.0, 10.0), "knife", 0.5)],
        PRIOR, VOCAB_V, VOCAB_N, P1,
    )  # fmt: skip
    assert len(props) == 1
    assert props[0].verbs == (("take", 0.8), ("cut", 0.6))


def test_pairing_confident_verb_without_noun_does_not_block_its_neighbour():
    # Review of #18: take (0, 10) 0.9 and cut (2, 12) 0.5 overlap (tIoU 8/12 = .67).
    # knife (4, 14) matches cut (8/12 = .67) but not take (6/14 = .43). take can't anchor
    # (no noun); cut must still produce the action. The old "top verb of its group" rule
    # skipped cut because take led the group, and returned nothing.
    props = pair_proposals(
        [((0.0, 10.0), "take", 0.9), ((2.0, 12.0), "cut", 0.5)],
        [((4.0, 14.0), "knife", 0.8)],
        PRIOR, VOCAB_V, VOCAB_N, P1,
    )  # fmt: skip
    assert [pr.segment for pr in props] == [(2.0, 12.0)]


def test_pairing_chain_with_nounless_middle_verb_keeps_the_action():
    # X (0, 10) 0.9 – Y (3, 13) 0.8 – Z (6, 16) 0.5: X–Y and Y–Z overlap (tIoU 7/13 = .54),
    # X–Z don't (4/16 = .25). Only Z matches onion (7, 17) (9/11 = .82; Y 6/14 = .43).
    # Z anchors (nothing anchored before it overlaps it) → one proposal on (6, 16).
    props = pair_proposals(
        [((0.0, 10.0), "take", 0.9), ((3.0, 13.0), "cut", 0.8), ((6.0, 16.0), "take", 0.5)],
        [((7.0, 17.0), "onion", 0.7)],
        PRIOR, VOCAB_V, VOCAB_N, P1,
    )  # fmt: skip
    assert [pr.segment for pr in props] == [(6.0, 16.0)]


def test_pairing_identical_duplicate_detections_give_one_proposal():
    dets = [((0.0, 10.0), "take", 0.8), ((0.0, 10.0), "take", 0.8)]
    props = pair_proposals(dets, [((0.0, 10.0), "knife", 0.5)], PRIOR, VOCAB_V, VOCAB_N, P1)
    assert len(props) == 1


def test_pairing_top_k_limits_pairs():
    k1 = PairingParams(match_tiou=0.5, top_k=1, alpha=1.0, eps=1.0, score_power=1.0)
    props = pair_proposals(
        [((0.0, 10.0), "take", 0.8), ((0.0, 10.0), "cut", 0.6)],
        [((0.0, 10.0), "knife", 0.5), ((0.0, 10.0), "onion", 0.4)],
        PRIOR, VOCAB_V, VOCAB_N, k1,
    )  # fmt: skip
    assert len(props[0].verbs) == len(props[0].nouns) == len(props[0].pairs) == 1


def test_pairing_empty_inputs():
    assert pair_proposals([], [((0.0, 1.0), "knife", 0.5)], PRIOR, VOCAB_V, VOCAB_N, P1) == []


# --- score rescaling ------------------------------------------------------------------


def test_score_power_is_monotone_and_video_independent():
    # The same (p_v, p_n, prior) must give the same score in any video, and raising
    # p_v must never lower the score.
    sq = PairingParams(match_tiou=0.5, top_k=5, alpha=0.5, eps=1.0, score_power=0.5)

    def score(pv, extra_dets=()):
        dets = [((0.0, 10.0), "take", pv), *extra_dets]
        props = pair_proposals(dets, [((0.0, 10.0), "knife", 0.5)], PRIOR, VOCAB_V, VOCAB_N, sq)
        return next(s for pr in props for v, n, s in pr.pairs if pr.segment == (0.0, 10.0))

    # (0.8 * 0.5 * (4/6) ** 0.5) ** 0.5 = (0.4 * 0.816497) ** 0.5 = 0.326599 ** 0.5 = 0.571488
    assert score(0.8) == pytest.approx(0.571488, abs=1e-6)
    other_video = [((50.0, 60.0), "cut", 0.99)]
    assert score(0.8, other_video) == pytest.approx(score(0.8), abs=1e-12)
    values = [score(p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert values == sorted(values)


# --- proposals_to_actions -------------------------------------------------------------


def _props():
    return pair_proposals(
        [((0.0, 10.0), "take", 0.8), ((20.0, 30.0), "cut", 0.9)],
        [((0.0, 10.0), "knife", 0.5), ((20.0, 30.0), "onion", 0.6)],
        PRIOR, VOCAB_V, VOCAB_N, P1,
    )  # fmt: skip


def test_actions_sorted_by_score_and_in_unit_interval():
    # cut-onion = 0.9 * 0.6 * 5/6 = 0.45; take-knife = 0.8 * 0.5 * 4/6 = 0.26667
    actions = proposals_to_actions(_props(), max_per_video=10, min_score=0.0)
    assert [(a[1], a[2]) for a in actions] == [("cut", "onion"), ("take", "knife")]
    assert all(0.0 <= a[3] <= 1.0 for a in actions)


def test_actions_capped_and_filtered():
    assert len(proposals_to_actions(_props(), max_per_video=1, min_score=0.0)) == 1
    # min_score 0.3 drops take-knife (0.2667)
    kept = proposals_to_actions(_props(), max_per_video=10, min_score=0.3)
    assert [(a[1], a[2]) for a in kept] == [("cut", "onion")]


def test_actions_cap_above_kit_limit_rejected():
    with pytest.raises(ValueError):
        proposals_to_actions(_props(), max_per_video=2001, min_score=0.0)


def test_pairing_long_zero_shot_video_stays_small_in_memory():
    # Regression (#21 review): zero-shot detections on a 2000 s video (~6000 verb and
    # ~6000 noun windows) made dense 6000 x 6000 tIoU matrices: ~1 GB and ~8 s. Only
    # overlapping pairs matter, so peak memory must stay far below that.
    import tracemalloc

    rng = np.random.default_rng(0)
    n = 6000
    starts = np.sort(rng.uniform(0, 1998, n))
    segs = [(float(s), float(s) + 2.0) for s in starts]
    verbs = [(seg, "take", float(p)) for seg, p in zip(segs, rng.uniform(0, 1, n), strict=True)]
    nouns = [(seg, "knife", float(p)) for seg, p in zip(segs, rng.uniform(0, 1, n), strict=True)]

    tracemalloc.start()
    props = pair_proposals(verbs, nouns, PRIOR, VOCAB_V, VOCAB_N, P1)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert props  # the work was actually done
    assert peak < 100 * 2**20  # dense matrices alone would be ~860 MB


def test_class_soft_nms_off_returns_actions_unchanged():
    from rewind.detect.pairing import class_soft_nms

    acts = [((0.0, 2.0), "take", "cup", 0.9), ((0.5, 2.5), "take", "cup", 0.8)]
    assert class_soft_nms(acts, 0.0) == acts


def test_class_soft_nms_decays_same_class_overlaps_only():
    from rewind.detect.pairing import class_soft_nms

    # take/cup (0, 2) 0.9 and (1, 3) 0.8: tIoU 1/3 -> 0.8 * exp(-(1/9) / 0.5) = 0.8 * 0.80074
    # = 0.64059. put/cup (1, 3) 0.7 is another class: untouched.
    acts = [((0.0, 2.0), "take", "cup", 0.9), ((1.0, 3.0), "take", "cup", 0.8),
            ((1.0, 3.0), "put", "cup", 0.7)]  # fmt: skip
    out = class_soft_nms(acts, 0.5)
    scores = {(a[0], a[1]): a[3] for a in out}
    assert scores[((0.0, 2.0), "take")] == pytest.approx(0.9)
    assert scores[((1.0, 3.0), "take")] == pytest.approx(0.8 * math.exp(-(1 / 9) / 0.5))
    assert scores[((1.0, 3.0), "put")] == pytest.approx(0.7)
    assert [a[3] for a in out] == sorted((a[3] for a in out), reverse=True)


def test_pair_proposals_keep_widens_the_pairs_per_proposal():
    # one verb segment with 2 verbs x 2 nouns = 4 pairs; top_k 2 caps at 2 unless keep=4
    verbs = [((0.0, 4.0), "take", 0.9), ((0.0, 4.0), "put", 0.5)]
    nouns = [((0.0, 4.0), "cup", 0.8), ((0.0, 4.0), "plate", 0.4)]
    params = PairingParams(match_tiou=0.5, top_k=2, alpha=0.0, eps=1.0, score_power=1.0)
    prior = VNPrior.from_counts(("take", "put"), ("cup", "plate"), np.ones((2, 2)), eps=1.0)
    voc_v, voc_n = {"take", "put"}, {"cup", "plate"}
    (p2,) = pair_proposals(verbs, nouns, prior, voc_v, voc_n, params)
    (p4,) = pair_proposals(verbs, nouns, prior, voc_v, voc_n, params, keep=4)
    assert len(p2.pairs) == 2 and len(p4.pairs) == 4


def test_pair_proposals_noun_tiou_relaxes_only_the_verb_noun_gate():
    # verb (0, 4) and noun (2, 6): tIoU 2 / 6 = 0.33. Gate 0.5 -> no proposal;
    # noun_tiou 0.3 -> one proposal on the verb's segment.
    verbs = [((0.0, 4.0), "take", 0.9)]
    nouns = [((2.0, 6.0), "cup", 0.8)]
    params = PairingParams(match_tiou=0.5, top_k=5, alpha=0.0, eps=1.0, score_power=1.0)
    prior = VNPrior.from_counts(("take",), ("cup",), np.ones((1, 1)), eps=1.0)
    assert pair_proposals(verbs, nouns, prior, {"take"}, {"cup"}, params) == []
    (p,) = pair_proposals(verbs, nouns, prior, {"take"}, {"cup"}, params, noun_tiou=0.3)
    assert p.segment == (0.0, 4.0)
