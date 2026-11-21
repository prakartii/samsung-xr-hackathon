"""cluster_proposals: linking rules and pair distributions (ARCHITECTURE.md §5.2)."""

import numpy as np
import pytest

from rewind.cluster import calibrate, cluster_proposals
from rewind.types import Proposal

KW = {"tiou_thr": 0.3, "sim_thr": 0.85, "k": 5, "temperature": 1.0}


def prop(s, e, *pairs):
    return Proposal((s, e), (), (), tuple(pairs))


def test_cluster_overlapping_same_top1_merge_with_union_span():
    # (0, 10) and (2, 12): tIoU 8/12 = .67 ≥ .3, both top-1 cut/onion → one cluster.
    cl = cluster_proposals(
        [prop(0, 10, ("cut", "onion", 0.9)), prop(2, 12, ("cut", "onion", 0.5))], **KW
    )

    assert len(cl) == 1
    assert cl[0].span == (0, 12)
    assert cl[0].member_idx == (0, 1)


def test_cluster_overlapping_different_top1_stay_apart_without_embeddings():
    cl = cluster_proposals(
        [prop(0, 10, ("cut", "onion", 0.9)), prop(2, 12, ("wash", "knife", 0.9))], **KW
    )

    assert len(cl) == 2


def test_cluster_different_top1_merge_when_embeddings_are_similar():
    emb = np.array([[1.0, 0.0], [0.9, np.sqrt(1 - 0.81)]])  # cosine 0.9 ≥ 0.85
    props = [prop(0, 10, ("cut", "onion", 0.9)), prop(2, 12, ("wash", "knife", 0.9))]

    assert len(cluster_proposals(props, embeddings=emb, **KW)) == 1


def test_cluster_low_tiou_same_pair_stays_apart():
    # (0, 10) and (8, 20): tIoU 2/20 = .1 < .3
    cl = cluster_proposals(
        [prop(0, 10, ("cut", "onion", 0.9)), prop(8, 20, ("cut", "onion", 0.9))], **KW
    )

    assert len(cl) == 2


def test_cluster_links_are_transitive():
    # A (0, 10)–B (5, 15): 5/15 = .33; B–C (10, 18): 5/13 = .38; A–C don't overlap.
    p = ("cut", "onion", 0.9)
    cl = cluster_proposals([prop(0, 10, p), prop(5, 15, p), prop(10, 18, p)], **KW)

    assert [c.member_idx for c in cl] == [(0, 1, 2)]


def test_cluster_pair_dist_is_score_weighted_mix():
    # A: (cut onion .6, cut carrot .2), sum .8, weight .6 → onion .45, carrot .15.
    # B: (cut onion .9), weight .9 → onion .9.  Mass: onion 1.35, carrot .15 → .9 / .1.
    a = prop(0, 10, ("cut", "onion", 0.6), ("cut", "carrot", 0.2))
    b = prop(1, 10, ("cut", "onion", 0.9))

    (c,) = cluster_proposals([a, b], **KW)

    assert dict(c.pair_dist) == pytest.approx(
        {("cut", "onion"): 0.9, ("cut", "carrot"): 0.1}, abs=1e-12
    )
    assert c.confidence == pytest.approx(0.9, abs=1e-12)


def test_cluster_pair_dist_truncated_to_k():
    a = prop(0, 10, ("cut", "onion", 0.6), ("cut", "carrot", 0.2))

    (c,) = cluster_proposals([a], **{**KW, "k": 1})

    assert dict(c.pair_dist) == pytest.approx({("cut", "onion"): 1.0}, abs=1e-12)


def test_calibrate_temperature_sharpens():
    # T = 0.5 → p ∝ m²: 1.35² = 1.8225, .15² = .0225 → .98780, .01220.
    p = calibrate({("a", "x"): 1.35, ("b", "x"): 0.15}, 0.5)

    assert p == pytest.approx({("a", "x"): 1.8225 / 1.845, ("b", "x"): 0.0225 / 1.845}, abs=1e-12)


def test_cluster_ids_follow_span_start():
    p, q = ("cut", "onion", 0.9), ("wash", "knife", 0.9)
    cl = cluster_proposals([prop(50, 60, p), prop(0, 5, q)], **KW)

    assert [(c.cluster_id, c.span) for c in cl] == [(0, (0, 5)), (1, (50, 60))]


def test_cluster_pair_dist_is_read_only():
    (c,) = cluster_proposals([prop(0, 10, ("cut", "onion", 0.9))], **KW)

    with pytest.raises(TypeError):
        c.pair_dist[("x", "y")] = 1.0


def test_cluster_confidence_is_the_max_member_score_in_any_order():
    # The more confident member comes first: confidence must still be 0.9, not the last.
    a = prop(0, 10, ("cut", "onion", 0.9))
    b = prop(1, 10, ("cut", "onion", 0.4))

    (c,) = cluster_proposals([a, b], **KW)

    assert c.confidence == pytest.approx(0.9, abs=1e-12)


CHAIN = [
    # (0, 10)-(2, 12): tIoU 8/12 = .67; (2, 12)-(4, 14): .67; all top-1 cut/onion -> one cluster
    prop(0, 10, ("cut", "onion", 0.9)),
    prop(2, 12, ("cut", "onion", 0.8)),
    prop(4, 14, ("cut", "onion", 0.7)),
]


def test_cluster_span_mode_union_is_the_full_extent():
    (c,) = cluster_proposals(CHAIN, **KW)
    assert c.span == (0, 14)


def test_cluster_span_mode_median_is_median_start_to_median_end():
    # starts 0, 2, 4 -> median 2; ends 10, 12, 14 -> median 12
    (c,) = cluster_proposals(CHAIN, span_mode="median", **KW)
    assert c.span == (2, 12)


def test_cluster_span_mode_median_single_member_keeps_its_segment():
    (c,) = cluster_proposals([prop(3, 7, ("cut", "onion", 0.9))], span_mode="median", **KW)
    assert c.span == (3, 7)
