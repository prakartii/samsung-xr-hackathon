"""Selector invariants for any FPVSum-like input (CLAUDE.md §7.2)."""

from itertools import pairwise

from hypothesis import given, settings
from hypothesis import strategies as st

from rewind.select.importance import ImportanceParams, select_segments
from tests.strategies import dense_signal, durations

P = ImportanceParams(1.0, 0.5, 1.0, 1.0, 2.0, 0.15, 4.05, 10.0, 1.0, 0.5)
BUDGET = 0.145


@settings(max_examples=40, deadline=None)
@given(data=st.data(), duration=st.floats(min_value=5.0, max_value=400.0))
def test_selected_segments_satisfy_highlight_rules(data, duration):
    t, imp = data.draw(dense_signal(duration))
    n_cuts = data.draw(st.integers(min_value=0, max_value=5))
    cuts = sorted(
        data.draw(st.lists(st.floats(0.5, duration - 0.5), min_size=n_cuts, max_size=n_cuts))
    )
    bounds = [0.0, *cuts, duration]
    shots = [(a, b) for a, b in pairwise(bounds) if b > a]
    budget = BUDGET * duration
    segs = select_segments(t, imp, shots, duration, budget, P)

    assert segs == sorted(segs)
    assert all(a[1] <= b[0] + 1e-9 for a, b in pairwise(segs))  # non-overlapping
    assert all(0.0 <= s < e <= duration + 1e-9 for s, e in segs)  # inside [0, T]
    assert all(e - s >= 4.05 - 1e-9 for s, e in segs)  # kit 4 s rule (with margin)
    if budget >= P.min_window:
        assert sum(e - s for s, e in segs) <= budget + 1e-9
        assert segs, "a video with budget >= min_window must get at least one segment"
    elif duration >= P.min_window:
        assert len(segs) == 1  # short video: exactly one min_window window


@settings(max_examples=10, deadline=None)
@given(duration=durations)
def test_durations_strategy_in_range(duration):
    assert 5.0 <= duration <= 2000.0
