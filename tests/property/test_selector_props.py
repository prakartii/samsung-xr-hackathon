"""Selector invariants and the coverage function's monotone / diminishing returns."""

import time
from itertools import pairwise
from types import MappingProxyType

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rewind.select.coverage import expected_coverage, select_windows
from rewind.types import Cluster

PAIRS = [(v, n) for v in "abcde" for n in "xyz"]
KW = {"rule": "any_overlap", "theta": 0.5, "tau": 0.05}


@st.composite
def clusters(draw, duration=300.0, max_n=40, max_len=15.0):
    n = draw(st.integers(0, max_n))
    out = []
    for i in range(n):
        s = draw(st.floats(0.0, duration - 1.0))
        e = draw(st.floats(s + 0.1, min(duration, s + max_len)))
        k = draw(st.integers(1, 3))
        chosen = draw(st.lists(st.sampled_from(PAIRS), min_size=k, max_size=k, unique=True))
        w = draw(st.lists(st.floats(0.05, 1.0), min_size=k, max_size=k))
        z = sum(w)
        out.append(
            Cluster(
                i,
                (s, e),
                (i,),
                MappingProxyType({p: x / z for p, x in zip(chosen, w, strict=True)}),
                1.0,
            )
        )
    return out


segments = st.lists(
    st.tuples(st.floats(0.0, 295.0), st.just(4.05)).map(lambda t: (t[0], t[0] + t[1])), max_size=8
)


@settings(max_examples=300, deadline=None)
@given(
    clusters(max_len=80.0),
    st.floats(0.0, 60.0),
    st.sampled_from([0.5, 3.0, 8.0]),
    st.sampled_from(["any_overlap", "fraction"]),
)
def test_select_windows_output_is_valid_and_consistent(cs, budget, gap, rule):
    # Large merge gaps fill uncovered stretches (clusters there get captured); long
    # clusters under the fraction rule span several segments. Either way the incremental
    # state must match a from-scratch evaluation.
    kw = {**KW, "rule": rule}
    sel = select_windows(
        cs, 300.0, budget=budget, lengths=[4.05, 5.0, 6.0], stride=0.5, merge_gap=gap, **kw
    )

    assert all(0.0 <= s < e <= 300.0 for s, e in sel.segments)
    assert all(e - s >= 4.05 - 1e-9 for s, e in sel.segments)
    assert all(s1 - e0 >= gap - 1e-9 for (_, e0), (s1, _) in pairwise(sel.segments))
    assert sum(e - s for s, e in sel.segments) <= budget + 1e-9
    # The incremental state matches a from-scratch evaluation of the result.
    assert abs(sel.expected - expected_coverage(cs, sel.segments, **kw)) < 1e-9


@settings(max_examples=150, deadline=None)
@given(clusters(), segments, segments, st.tuples(st.floats(0.0, 295.0)))
def test_expected_coverage_monotone_and_diminishing_returns(cs, a, extra, w_start):
    # A ⊆ B = A + extra; adding w helps A at least as much as it helps B.
    w = (w_start[0], w_start[0] + 4.05)
    b = a + extra
    e = lambda segs: expected_coverage(cs, segs, **KW)  # noqa: E731

    assert e(b) >= e(a) - 1e-9  # monotone
    assert e([*a, w]) - e(a) >= e([*b, w]) - e(b) - 1e-9  # diminishing returns


@pytest.mark.slow
def test_select_windows_is_fast_on_a_long_video():
    # 2000 s video with 300 clusters must select in < 2 s (PROGRESS.md P5).
    cs = [
        Cluster(
            i, (i * 6.5, i * 6.5 + 3.0), (i,), MappingProxyType({PAIRS[i % len(PAIRS)]: 1.0}), 1.0
        )
        for i in range(300)
    ]
    t0 = time.perf_counter()

    select_windows(
        cs, 2000.0, budget=290.0, lengths=[4.05, 5.0, 6.0], stride=0.5, merge_gap=0.5, **KW
    )

    assert time.perf_counter() - t0 < 2.0


@pytest.mark.slow
def test_select_windows_is_fast_on_realistic_detector_output():
    # Review of #15: 2000 proposals (the actions cap) with top-5 pairs on a 2000 s video.
    import random

    from rewind.cluster import cluster_proposals
    from rewind.types import Proposal

    rng = random.Random(0)
    pairs = [(f"v{i}", f"n{j}") for i in range(30) for j in range(40)]
    props = []
    for _ in range(2000):
        s = rng.uniform(0, 1999)
        top = rng.sample(pairs, 5)
        w = sorted((rng.random() for _ in top), reverse=True)
        scored = tuple((v, n, x) for (v, n), x in zip(top, w, strict=True))
        props.append(Proposal((s, min(2000.0, s + rng.uniform(0.5, 8))), (), (), scored))
    cs = cluster_proposals(props, tiou_thr=0.3, sim_thr=0.85, k=5, temperature=1.0)
    t0 = time.perf_counter()

    select_windows(
        cs, 2000.0, budget=290.0, lengths=[4.05, 5.0, 6.0], stride=0.5, merge_gap=0.5, **KW
    )

    assert time.perf_counter() - t0 < 2.0
