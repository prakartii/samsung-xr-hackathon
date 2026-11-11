"""Candidate windows, capture model and expected-coverage selection (§5.3–§5.4)."""

import math
from types import MappingProxyType

import pytest

from rewind.select.candidates import candidate_windows, capture_prob
from rewind.select.coverage import expected_coverage, merge_union, select_windows
from rewind.types import Cluster


def cl(cid, span, dist):
    return Cluster(cid, span, (cid,), MappingProxyType(dist), max(dist.values()))


A, B, C = ("cut", "onion"), ("wash", "knife"), ("open", "fridge")
SEL = {"stride": 0.5, "merge_gap": 0.5, "rule": "any_overlap", "theta": 0.5, "tau": 0.05}


def test_candidate_windows_overlap_the_span():
    # Span (10, 11), 4.05 s windows, 1 s grid: start < 11 and start + 4.05 > 10 → 6..10.
    w = candidate_windows(20.0, [4.05], 1.0, [(10.0, 11.0)])

    assert [s for s, _ in w] == [6.0, 7.0, 8.0, 9.0, 10.0]


def test_candidate_windows_stay_inside_the_video():
    # Span (19, 20) in T = 20: start ≤ 20 - 4.05 = 15.95 and > 14.95 → only 15.
    assert candidate_windows(20.0, [4.05], 1.0, [(19.0, 20.0)]) == [(15.0, 19.05)]


def test_candidate_windows_longer_than_video_are_skipped():
    assert candidate_windows(5.0, [6.0], 0.5, [(1.0, 2.0)]) == []


def test_capture_prob_rules():
    assert capture_prob(0.0, 2.0, "any_overlap", 0.5, 0.05) == 0.0
    assert capture_prob(0.1, 2.0, "any_overlap", 0.5, 0.05) == 1.0
    # fraction: covered 1 of 2 s = θ → sigmoid(0) = 0.5; all of it → sigmoid(10).
    assert capture_prob(1.0, 2.0, "fraction", 0.5, 0.05) == pytest.approx(0.5, abs=1e-12)
    assert capture_prob(2.0, 2.0, "fraction", 0.5, 0.05) == pytest.approx(
        1 / (1 + math.exp(-10)), abs=1e-12
    )


def test_merge_union_matches_postprocess_gap_rule():
    # gap 0.3 < 0.5 merges; gap exactly 0.5 stays separate (as in postprocess).
    assert merge_union([(0.0, 5.0)], (5.3, 10.0), 0.5) == [(0.0, 10.0)]
    assert merge_union([(0.0, 5.0)], (5.5, 10.0), 0.5) == [(0.0, 5.0), (5.5, 10.0)]


def test_expected_coverage_hand_computed():
    # c1 {A .5, B .5} at (0, 2); c2 {A 1} at (10, 12).
    clusters = [cl(0, (0.0, 2.0), {A: 0.5, B: 0.5}), cl(1, (10.0, 12.0), {A: 1.0})]
    kw = {"rule": "any_overlap", "theta": 0.5, "tau": 0.05}

    # Covering c1 only: P_miss(A) = .5, P_miss(B) = .5 → E = .5 + .5 = 1.0.
    assert expected_coverage(clusters, [(0.0, 4.05)], **kw) == pytest.approx(1.0, abs=1e-12)
    # Adding c2: P_miss(A) = .5 · 0 = 0 → E = 1 + .5 = 1.5.
    assert expected_coverage(clusters, [(0.0, 4.05), (10.0, 14.05)], **kw) == pytest.approx(
        1.5, abs=1e-12
    )


def test_select_one_window_covers_two_pairs_and_skips_a_repeat():
    # A at (10, 11), B at (12, 13), A again at (50, 51). Budget 8.1 s (two windows).
    # First pick captures A and B: starts in (7.95, 11), earliest 8.0 → (8.0, 12.05).
    # The repeat of A adds nothing, so the second window is never spent.
    clusters = [
        cl(0, (10.0, 11.0), {A: 1.0}),
        cl(1, (12.0, 13.0), {B: 1.0}),
        cl(2, (50.0, 51.0), {A: 1.0}),
    ]

    sel = select_windows(clusters, 100.0, budget=8.1, lengths=[4.05], **SEL)

    assert sel.segments == ((8.0, 12.05),)
    assert sel.expected == pytest.approx(2.0, abs=1e-12)


def test_select_prefers_a_new_pair_over_an_uncertain_repeat():
    # c1 {A} at (10, 11), c2 {A .5, B .5} at (30, 31), c3 {C} at (60, 61); two windows.
    # All start with gain 1; c1's window is earliest. Then c2 only adds B with p .5
    # (gain .5) while c3 adds C (gain 1) → second pick is c3's earliest window.
    clusters = [
        cl(0, (10.0, 11.0), {A: 1.0}),
        cl(1, (30.0, 31.0), {A: 0.5, B: 0.5}),
        cl(2, (60.0, 61.0), {C: 1.0}),
    ]

    sel = select_windows(clusters, 100.0, budget=8.1, lengths=[4.05], **SEL)

    assert sel.segments == ((6.0, 10.05), (56.0, 60.05))
    assert sel.expected == pytest.approx(2.0, abs=1e-12)


def test_select_longer_window_wins_when_it_captures_more_per_second():
    # A at (10, 10.2), B at (14.3, 14.5): no 4.05 s window reaches both, but a 6 s one
    # starting in (8.3, 10.2) does: 2 pairs / 6 s = .333/s beats 1 / 4.05 = .247/s.
    # Earliest such start on the 0.5 s grid: 8.5 → (8.5, 14.5).
    clusters = [cl(0, (10.0, 10.2), {A: 1.0}), cl(1, (14.3, 14.5), {B: 1.0})]

    sel = select_windows(clusters, 100.0, budget=6.0, lengths=[4.05, 6.0], **SEL)

    assert sel.segments == ((8.5, 14.5),)
    assert sel.expected == pytest.approx(2.0, abs=1e-12)


def test_select_merge_cost_includes_the_filled_gap():
    # A at (10, 10.2), B at (14.4, 14.5); budget 8.5. First pick (6.0, 10.05) for A.
    # For B, (10.5, 14.55) sits 0.45 s away, so it merges: cost 14.55 - 10.05 = 4.5,
    # total 8.55 > 8.5 → doesn't fit. (11.0, 15.05) stands alone: cost 4.05, total 8.1.
    clusters = [cl(0, (10.0, 10.2), {A: 1.0}), cl(1, (14.4, 14.5), {B: 1.0})]

    sel = select_windows(clusters, 100.0, budget=8.5, lengths=[4.05], **SEL)

    assert sel.segments == ((6.0, 10.05), (11.0, 15.05))


def test_select_respects_zero_budget():
    sel = select_windows([cl(0, (10.0, 11.0), {A: 1.0})], 100.0, budget=0.0, lengths=[4.05], **SEL)

    assert sel.segments == ()
