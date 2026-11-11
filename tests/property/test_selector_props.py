"""Selector invariants and the coverage function's monotone / diminishing returns."""

import time
from itertools import pairwise
from types import MappingProxyType

from hypothesis import given, settings
from hypothesis import strategies as st

from rewind.select.coverage import expected_coverage, select_windows
from rewind.types import Cluster

PAIRS = [(v, n) for v in "abcde" for n in "xyz"]
KW = {"rule": "any_overlap", "theta": 0.5, "tau": 0.05}


@st.composite
def clusters(draw, duration=300.0, max_n=40):
    n = draw(st.integers(0, max_n))
    out = []
    for i in range(n):
        s = draw(st.floats(0.0, duration - 1.0))
        e = draw(st.floats(s + 0.1, min(duration, s + 15.0)))
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


@settings(max_examples=150, deadline=None)
@given(clusters(), st.floats(0.0, 60.0))
def test_select_windows_output_is_valid_and_consistent(cs, budget):
    sel = select_windows(
        cs, 300.0, budget=budget, lengths=[4.05, 5.0, 6.0], stride=0.5, merge_gap=0.5, **KW
    )

    assert all(0.0 <= s < e <= 300.0 for s, e in sel.segments)
    assert all(e - s >= 4.05 - 1e-9 for s, e in sel.segments)
    assert all(s1 - e0 >= 0.5 - 1e-9 for (_, e0), (s1, _) in pairwise(sel.segments))
    assert sum(e - s for s, e in sel.segments) <= budget + 1e-9
    # The incremental state matches a from-scratch evaluation of the result.
    assert abs(sel.expected - expected_coverage(cs, sel.segments, **KW)) < 1e-9


@settings(max_examples=150, deadline=None)
@given(clusters(), segments, segments, st.tuples(st.floats(0.0, 295.0)))
def test_expected_coverage_monotone_and_diminishing_returns(cs, a, extra, w_start):
    # A ⊆ B = A + extra; adding w helps A at least as much as it helps B.
    w = (w_start[0], w_start[0] + 4.05)
    b = a + extra
    e = lambda segs: expected_coverage(cs, segs, **KW)  # noqa: E731

    assert e(b) >= e(a) - 1e-9  # monotone
    assert e([*a, w]) - e(a) >= e([*b, w]) - e(b) - 1e-9  # diminishing returns


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
