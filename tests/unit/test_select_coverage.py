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


# --- review fixes (PR #15) ------------------------------------------------------


def test_select_refreshes_windows_next_to_a_pick():
    # Review counterexample (T = 60, budget 10, windows 4.05/5/6). Exact greedy reaches
    # E = 4.19 with (14.0, 19.55) and (39.5, 43.55); without re-scoring the windows next
    # to each pick, lazy greedy got stuck at (14.0, 20.0) with E = 2.82.
    bx, az, by, bz, ax = ("b", "x"), ("a", "z"), ("b", "y"), ("b", "z"), ("a", "x")
    cs = [
        cl(0, (41.73164349674044, 42.43692632367619), {bx: 1.0}),
        cl(1, (19.391035784961026, 20.446903132139127), {bx: 0.6272, az: 0.3728}),
        cl(2, (18.23540441877319, 20.865554309026862), {by: 0.8209, bz: 0.1791}),
        cl(3, (58.973937967211754, 60.0), {az: 0.4091, ax: 0.5909}),
        cl(4, (27.526684172830002, 28.185507657374448), {bz: 1.0}),
        cl(5, (43.253991900805715, 45.6926693726914), {ax: 1.0}),
        cl(6, (11.198987270633276, 14.077557123934433), {bz: 1.0}),
    ]

    sel = select_windows(cs, 60.0, budget=10.0, lengths=[4.05, 5.0, 6.0], **SEL)

    assert sel.expected >= 4.19 - 1e-2


def test_select_values_count_expected_captured_clusters_per_segment():
    # One window over A and B: both clusters captured with probability 1 → value 2.
    clusters = [cl(0, (10.0, 11.0), {A: 1.0}), cl(1, (12.0, 13.0), {B: 1.0})]

    sel = select_windows(clusters, 100.0, budget=4.05, lengths=[4.05], **SEL)

    assert sel.values == (2.0,)


def test_candidate_windows_touching_the_span_are_excluded():
    # 4.0 s windows on a 0.25 s grid around (10, 11): start 6.0 ends exactly at 10
    # (touching, zero overlap) → excluded; the first is 6.25.
    w = candidate_windows(20.0, [4.0], 0.25, [(10.0, 11.0)])

    assert w[0] == (6.25, 10.25)


def test_select_counts_clusters_inside_a_filled_gap():
    # merge_gap 100: every window merges, so the union fills the stretch between picks.
    # The cluster at (20, 20.1) repeats pair a (no gain of its own) and is captured only
    # through the filled gap; its capture must still be recorded: value 3, not 2.
    a, b = ("a", "x"), ("b", "x")
    cs = [
        cl(0, (10.0, 10.1), {a: 1.0}),
        cl(1, (20.0, 20.1), {a: 1.0}),
        cl(2, (30.0, 30.1), {b: 1.0}),
    ]
    kw = {**SEL, "merge_gap": 100.0}

    sel = select_windows(cs, 100.0, budget=40.0, lengths=[4.05], **kw)

    assert sel.segments == ((6.0, 30.05),)
    assert sel.values == (3.0,)


def test_select_fraction_rule_state_matches_from_scratch_on_a_long_cluster():
    # One long cluster (0, 20) under the fraction rule: its capture grows with every
    # window added over it (they end up merged into (0, 12.05)). The incremental state
    # must equal a from-scratch evaluation of the chosen segments.
    cs = [cl(0, (0.0, 20.0), {A: 1.0})]
    kw = {**SEL, "rule": "fraction"}

    sel = select_windows(cs, 40.0, budget=12.15, lengths=[4.05], **kw)

    assert sel.expected == pytest.approx(
        expected_coverage(cs, sel.segments, rule="fraction", theta=0.5, tau=0.05), abs=1e-9
    )


def test_candidate_windows_never_end_past_the_video():
    # T just under 2000 by float noise: no window may end at 2000.0 > T.
    T = 1999.9999999999998

    assert all(e <= T for _, e in candidate_windows(T, [5.0, 6.0], 1.0, [(1990.0, T)]))
