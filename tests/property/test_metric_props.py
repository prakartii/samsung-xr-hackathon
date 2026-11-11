"""Metric invariants: in [0, 1]; perfect → 1; empty → 0; more highlights never hurt."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from rewind_eval.coverage import coverage
from rewind_eval.f1 import frame_f1
from rewind_eval.map import mean_ap

VERBS, NOUNS = ["cut", "open", "wash"], ["onion", "fridge", "knife"]


@st.composite
def gt_actions(draw, duration=200.0):
    n = draw(st.integers(1, 30))
    out = []
    for _ in range(n):
        s = draw(st.floats(0.0, duration - 1.0))
        e = draw(st.floats(s + 0.1, min(duration, s + 30.0)))
        out.append(((s, e), draw(st.sampled_from(VERBS)), draw(st.sampled_from(NOUNS))))
    return out


@st.composite
def disjoint_highlights(draw, duration=200.0):
    starts = sorted(draw(st.sets(st.integers(0, int(duration) - 6), max_size=6)))
    out, last_end = [], -1.0
    for s in starts:
        if s > last_end:
            out.append((float(s), float(s) + 4.05))
            last_end = s + 4.05
    return out


@settings(max_examples=200, deadline=None)
@given(gt_actions(), disjoint_highlights(), st.sampled_from(["any_overlap", "fraction"]))
def test_coverage_is_in_unit_interval(gt, hl, rule):
    assert 0.0 <= coverage(gt, hl, 200.0, rule=rule) <= 1.0


@settings(max_examples=200, deadline=None)
@given(gt_actions(), disjoint_highlights(), disjoint_highlights())
def test_coverage_never_decreases_when_a_highlight_is_added_within_budget(gt, a, b):
    # 0.15 · 2000 s = 300 s of budget: no penalty for ≤ 12 short windows.
    merged = sorted(set(a) | set(b))
    before = coverage(gt, a, 2000.0, rule="any_overlap")
    after = coverage(gt, merged, 2000.0, rule="any_overlap")
    assert after >= before - 1e-12


@settings(max_examples=100, deadline=None)
@given(gt_actions())
def test_map_perfect_is_one_and_empty_is_zero(gt):
    gts = {"v": gt}
    perfect = {"v": [(s, v, n, 1.0) for s, v, n in gt]}
    assert abs(mean_ap(perfect, gts)["avg"] - 1.0) < 1e-9
    assert mean_ap({}, gts)["avg"] == 0.0


@settings(max_examples=100, deadline=None)
@given(st.lists(st.floats(0.0, 1.0), min_size=20, max_size=200), disjoint_highlights())
def test_frame_f1_is_in_unit_interval(imp, hl):
    times = np.arange(len(imp), dtype=float)
    assert 0.0 <= frame_f1(times, np.array(imp), hl) <= 1.0
