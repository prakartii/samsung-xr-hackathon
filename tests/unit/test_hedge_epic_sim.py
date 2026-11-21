"""Hedge windows, the EPIC selector entry point and the proposal simulator."""

import dataclasses
import itertools
from types import MappingProxyType

import numpy as np
import pytest

from rewind.config import load_config
from rewind.select.epic import select_epic, top1_clusters
from rewind.select.hedge import hedge_windows
from rewind.types import Cluster
from rewind_eval.sim import CLEAN, Noise, simulate
from tests.conftest import DEFAULT_CONFIG

KW = {"gap": 0.5, "strong": 0.8, "min_spacing": 10.0}


def cl(cid, span, dist, conf=1.0):
    return Cluster(cid, span, (cid,), MappingProxyType(dist), conf)


# --- hedge ---------------------------------------------------------------


def test_hedge_without_signals_or_selection_spreads_from_the_middle():
    # Farthest-point from the video ends {0, 100}: first the centre nearest 50 on the grid
    # 2.025 + 0.5k with the largest distance (50.025, 49.975 from 100) → (48.0, 52.05);
    # then the point farthest from {0, 50.025, 100}: 25.025 (25.0 away) → (23.0, 27.05).
    hl = hedge_windows([], [], 100.0, budget=8.1, **KW)

    assert hl == [(23.0, 27.05), (48.0, 52.05)]


def test_hedge_without_profile_goes_farthest_from_selected_and_ends():
    # Anchors: video ends 0 and 100 and the selected mid 5. The best grid centre is
    # 52.525 (47.525 from 5, 47.475 from 100) → (50.5, 54.55).
    assert hedge_windows([(0.0, 10.0)], [], 100.0, budget=4.05, **KW) == [(50.5, 54.55)]


def test_hedge_avoids_confident_clusters():
    # A confident cluster on (40, 60) blocks the middle. Valid centres: ≤ 37.525 (best
    # 37.525, 32.525 from 5) or ≥ 62.025 (37.975 from 100) → (60.0, 64.05).
    strong = [cl(0, (40.0, 60.0), {("a", "x"): 1.0}, conf=0.9)]

    assert hedge_windows([(0.0, 10.0)], strong, 100.0, budget=4.05, **KW) == [(60.0, 64.05)]


def test_hedge_follows_the_activity_profile():
    # Peak at 50; the nearest grid centre is 50.025 → (48.0, 52.05).
    profile = (np.array([0.0, 50.0, 100.0]), np.array([0.0, 1.0, 0.0]))

    assert hedge_windows([], [], 100.0, budget=4.05, profile=profile, **KW) == [(48.0, 52.05)]


def test_hedge_keeps_gap_from_selected():
    # Peak at 52 but (40, 50) is selected: windows must start ≥ 50.5 → centre 52.525.
    profile = (np.array([0.0, 52.0, 100.0]), np.array([0.0, 1.0, 0.0]))

    hl = hedge_windows([(40.0, 50.0)], [], 100.0, budget=4.05, profile=profile, **KW)

    assert hl == [(50.5, 54.55)]


def test_hedges_spread_over_a_long_video():
    # Review of #16: five hedges after a selection at (0, 10) used to land 10 s apart at
    # the far end. Now every pair of hedges is ≥ 100 s apart.
    hl = hedge_windows([(0.0, 10.0)], [], 1000.0, budget=5 * 4.05, **KW)
    mids = [(s + e) / 2 for s, e in hl]

    assert len(hl) == 5
    assert min(b - a for a, b in itertools.pairwise(mids)) > 100.0


def test_hedge_count_is_exact_at_a_multiple_of_the_window():
    # 3 · 4.05 = 12.149999… in floats; it must still give 3 hedges, not 2.
    assert len(hedge_windows([], [], 1000.0, budget=3 * 4.05, **KW)) == 3


def test_hedges_never_overlap_even_with_small_min_spacing():
    # min_spacing 1 s with a sharp activity peak: neighbouring grid centres score almost
    # the same, but hedges must still be ≥ one window + gap apart (no overlap, no touch).
    profile = (np.array([0.0, 50.0, 100.0]), np.array([0.0, 1.0, 0.0]))

    hl = hedge_windows(
        [], [], 100.0, budget=5 * 4.05, profile=profile, **{**KW, "min_spacing": 1.0}
    )

    assert all(s1 - e0 >= 0.5 - 1e-9 for (_, e0), (s1, _) in itertools.pairwise(hl))


def test_hedge_zero_budget_gives_nothing():
    assert hedge_windows([], [], 100.0, budget=4.0, **KW) == []


# --- select_epic -------------------------------------------------------------


def test_top1_clusters_collapse_to_most_likely_pair():
    c = cl(0, (0.0, 1.0), {("a", "x"): 0.3, ("b", "y"): 0.7})

    assert dict(top1_clusters([c])[0].pair_dist) == {("b", "y"): 1.0}


def test_select_epic_with_hedge_share_spends_reserve_on_hedges():
    # No clusters: everything left goes to hedges. B = 0.145 · 100 = 14.5 → 3 windows.
    cfg = load_config(DEFAULT_CONFIG)
    cfg = dataclasses.replace(cfg, hedge=dataclasses.replace(cfg.hedge, share=0.125))

    segs, values = select_epic([], 100.0, cfg)

    assert len(segs) == 3 and values == [0.0, 0.0, 0.0]


def test_select_epic_dedup_mode_ignores_second_choices():
    # c0 {A} at (5, 6), c1 {A .5, B .5} at (40, 41), c2 {A} at (80, 81); budget for two
    # windows (0.09 · 100 = 9 s), no hedge. First pick: all gain 1 → earliest (c0).
    # Expected: c1 still adds B with p .5 → second window on c1. Dedup: c1 collapses to
    # {A}, a repeat, like c2 → no gain anywhere → one coverage window (leftover budget
    # then goes to a hedge, value 0).
    # Windows below were worked out under any-overlap capture; this tests the mode, so
    # the capture rule is pinned rather than taken from the shipped config.
    cfg = load_config(DEFAULT_CONFIG)
    cfg = dataclasses.replace(
        cfg, select=dataclasses.replace(cfg.select, budget_frac=0.09, capture_rule="any_overlap")
    )
    cs = [
        cl(0, (5.0, 6.0), {("a", "x"): 1.0}),
        cl(1, (40.0, 41.0), {("a", "x"): 0.5, ("b", "y"): 0.5}),
        cl(2, (80.0, 81.0), {("a", "x"): 1.0}),
    ]

    def chosen(mode):
        segs, values = select_epic(cs, 100.0, cfg, mode=mode)
        return [seg for seg, v in zip(segs, values, strict=True) if v > 0]

    assert chosen("expected") == [(1.0, 5.05), (36.0, 40.05)]
    assert chosen("dedup") == [(1.0, 5.05)]


def test_select_epic_short_video_picks_the_best_single_window():
    # T = 20: budget 0.145 · 20 = 2.9 s < 4.05, but one 4.05 s window must still go where
    # the clusters are: two distinct pairs at (14, 15) and (15.5, 16) → one window over both.
    # Value 2.0 = two clusters each captured with probability exactly 1 (any-overlap;
    # the fraction rule's smooth step gives 0.99995 each), so the rule is pinned.
    cfg = load_config(DEFAULT_CONFIG)
    cfg = dataclasses.replace(
        cfg, select=dataclasses.replace(cfg.select, capture_rule="any_overlap")
    )
    cs = [cl(0, (14.0, 15.0), {("a", "x"): 1.0}), cl(1, (15.5, 16.0), {("b", "x"): 1.0})]

    segs, values = select_epic(cs, 20.0, cfg)

    assert len(segs) == 1 and values == [2.0]
    s, e = segs[0]
    assert e - s == pytest.approx(4.05, abs=1e-9) and s < 15.0 and e > 15.5


# --- simulator ---------------------------------------------------------------
GT = [((10.0, 12.0), "cut", "onion"), ((30.0, 35.0), "wash", "knife")]
POOL = [
    ("cut", "onion"),
    ("cut", "carrot"),
    ("wash", "knife"),
    ("wash", "plate"),
    ("open", "fridge"),
]


def test_simulate_clean_reproduces_ground_truth():
    props = simulate(GT, 100.0, CLEAN, POOL, seed=0)

    assert [(p.segment, p.pairs) for p in props] == [
        ((10.0, 12.0), (("cut", "onion", 1.0),)),
        ((30.0, 35.0), (("wash", "knife", 1.0),)),
    ]


def test_simulate_is_deterministic_per_seed():
    noise = Noise("n", miss=0.3, jitter=1.0, top1_acc=0.4, topk=3, fp_per_min=2.0)

    assert simulate(GT, 100.0, noise, POOL, seed=7) == simulate(GT, 100.0, noise, POOL, seed=7)


def test_simulate_false_positive_count_follows_rate():
    # miss = 1 drops all GT; 3 per minute over 100 s → round(5.0) = 5 proposals.
    noise = Noise("fp", miss=1.0, jitter=0.0, top1_acc=1.0, topk=2, fp_per_min=3.0)

    assert len(simulate(GT, 100.0, noise, POOL, seed=1)) == 5


def test_simulate_top1_accuracy_one_ranks_truth_first():
    noise = Noise("k", miss=0.0, jitter=0.0, top1_acc=1.0, topk=3, fp_per_min=0.0)

    props = simulate(GT, 100.0, noise, POOL, seed=3)

    assert [p.pairs[0][:2] for p in props] == [("cut", "onion"), ("wash", "knife")]
    assert all(len(p.pairs) == 3 for p in props)


def test_simulate_never_produces_inverted_or_out_of_range_segments():
    # Jitter near the edges used to give (100.3, 100.0) or (0.0, -0.82).
    gt = [((99.5, 100.0), "cut", "onion"), ((0.0, 0.4), "wash", "knife")]
    noise = Noise("edge", miss=0.0, jitter=2.0, top1_acc=1.0, topk=1, fp_per_min=0.0)

    for seed in range(200):
        for p in simulate(gt, 100.0, noise, POOL, seed=seed):
            assert 0.0 <= p.segment[0] < p.segment[1] <= 100.0


def test_simulate_scores_are_a_confidence_split_over_the_top_k():
    # Scores ∝ 0.5^rank, normalized, times a confidence in [0.3, 0.9]: they sum to ≤ 0.9.
    noise = Noise("k", miss=0.0, jitter=0.0, top1_acc=0.5, topk=4, fp_per_min=0.0)

    for p in simulate(GT, 100.0, noise, POOL, seed=5):
        scores = [x for _, _, x in p.pairs]
        assert all(x > 0 for x in scores) and 0.3 - 1e-9 <= sum(scores) <= 0.9 + 1e-9
        assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        # Earliest window touching (5, 6): (1.0, 5.05), only 0.05 s (5%) inside.
        ("any_overlap", [(1.0, 5.05)]),
        # Official rule needs >= 50% inside: the earliest window holding all of (5, 6)
        # is (2.0, 6.05) (cap 0.99995 vs 0.00012 for the grazing one).
        ("fraction", [(2.0, 6.05)]),
    ],
)
def test_select_epic_capture_rule_decides_whether_a_grazing_window_counts(rule, expected):
    # One action at (5, 6) in 100 s; budget 0.045 * 100 = 4.5 s holds one 4.05 s window
    # (5 and 6 s windows don't fit). No hedge.
    cfg = load_config(DEFAULT_CONFIG)
    cfg = dataclasses.replace(
        cfg,
        select=dataclasses.replace(cfg.select, budget_frac=0.045, capture_rule=rule),
        hedge=dataclasses.replace(cfg.hedge, share=0.0),
    )
    segs, values = select_epic([cl(0, (5.0, 6.0), {("a", "x"): 1.0})], 100.0, cfg)
    assert [seg for seg, v in zip(segs, values, strict=True) if v > 0] == expected


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        # A (5, 6) and B (10.5, 11.5), distinct pairs; 100 s, budget 0.09 * 100 = 9 s, only
        # 4.05 s windows (a longer one could hold both), stride 0.25, official rule. Earliest
        # window holding all of A: (2.0, 6.05). Gap 0.5: the
        # earliest holding all of B, (7.5, 11.55), sits 1.45 s after A's window -> kept apart.
        (0.5, [(2.0, 6.05), (7.5, 11.55)]),
        # Gap 2.0: (7.5, 11.55) would merge with A's window (cost 11.55 - 6.05 = 5.5 s, total
        # 9.55 > 9) -> the earliest window 2 s clear of 6.05 that holds all of B: (8.25, 12.3).
        (2.0, [(2.0, 6.05), (8.25, 12.3)]),
    ],
)
def test_select_epic_uses_the_epic_merge_gap(gap, expected):
    cfg = load_config(DEFAULT_CONFIG)
    cfg = dataclasses.replace(
        cfg,
        select=dataclasses.replace(
            cfg.select,
            budget_frac=0.09,
            stride=0.25,
            window_lengths=(4.05,),
            capture_rule="fraction",
            fraction_theta=0.55,
        ),
        hedge=dataclasses.replace(cfg.hedge, share=0.0),
        post=dataclasses.replace(cfg.post, merge_gap=0.5, merge_gap_epic=gap),
    )
    cs = [cl(0, (5.0, 6.0), {("a", "x"): 1.0}), cl(1, (10.5, 11.5), {("b", "y"): 1.0})]
    segs, values = select_epic(cs, 100.0, cfg)
    assert [seg for seg, v in zip(segs, values, strict=True) if v > 0] == expected
