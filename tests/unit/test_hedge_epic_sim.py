"""Hedge windows, the EPIC selector entry point and the proposal simulator."""

import dataclasses
from types import MappingProxyType

import numpy as np

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


def test_hedge_without_signals_or_selection_takes_earliest_spaced_centres():
    # All scores 0 → grid order. Centres 2.025, then the first ≥ 10 s away: 12.025.
    hl = hedge_windows([], [], 100.0, budget=8.1, **KW)

    assert hl == [(0.0, 4.05), (10.0, 14.05)]


def test_hedge_without_profile_goes_farthest_from_selected():
    # Selected (0, 10): the farthest centre is the last on the grid 2.025 + 0.5k that
    # fits (≤ 97.975), i.e. 97.525 → (95.5, 99.55).
    assert hedge_windows([(0.0, 10.0)], [], 100.0, budget=4.05, **KW) == [(95.5, 99.55)]


def test_hedge_avoids_confident_clusters():
    # A confident cluster on (90, 100) blocks the far end. The farthest centre whose
    # window ends ≤ 90 is 87.525 → (85.5, 89.55).
    strong = [cl(0, (90.0, 100.0), {("a", "x"): 1.0}, conf=0.9)]

    assert hedge_windows([(0.0, 10.0)], strong, 100.0, budget=4.05, **KW) == [(85.5, 89.55)]


def test_hedge_follows_the_activity_profile():
    # Peak at 50; the nearest grid centre is 50.025 → (48.0, 52.05).
    profile = (np.array([0.0, 50.0, 100.0]), np.array([0.0, 1.0, 0.0]))

    assert hedge_windows([], [], 100.0, budget=4.05, profile=profile, **KW) == [(48.0, 52.05)]


def test_hedge_keeps_gap_from_selected():
    # Peak at 52 but (40, 50) is selected: windows must start ≥ 50.5 → centre 52.525.
    profile = (np.array([0.0, 52.0, 100.0]), np.array([0.0, 1.0, 0.0]))

    hl = hedge_windows([(40.0, 50.0)], [], 100.0, budget=4.05, profile=profile, **KW)

    assert hl == [(50.5, 54.55)]


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
    cfg = load_config(DEFAULT_CONFIG)
    cfg = dataclasses.replace(cfg, select=dataclasses.replace(cfg.select, budget_frac=0.09))
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
