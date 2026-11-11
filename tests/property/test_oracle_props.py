"""The oracle is an upper reference: never below uniform windows, always valid."""

import itertools

from hypothesis import given, settings
from hypothesis import strategies as st

from rewind_eval.baselines import uniform_highlights
from rewind_eval.coverage import coverage
from rewind_eval.oracle import coverage_oracle


@st.composite
def video(draw):
    T = draw(st.floats(30.0, 400.0))
    n = draw(st.integers(1, 40))
    gt = []
    for _ in range(n):
        s = draw(st.floats(0.0, T - 0.5))
        e = draw(st.floats(s + 0.2, min(T, s + 20.0)))
        gt.append(((s, e), draw(st.sampled_from("abcdefgh")), draw(st.sampled_from("xyz"))))
    return T, gt


@settings(max_examples=150, deadline=None)
@given(video())
def test_coverage_oracle_is_valid_and_not_below_uniform(case):
    T, gt = case

    hl = coverage_oracle(gt, T, budget_frac=0.145)

    assert all(e - s >= 4.0 for s, e in hl)
    assert all(e0 < s1 for (_, e0), (s1, _) in itertools.pairwise(hl))
    assert sum(e - s for s, e in hl) <= 0.145 * T + 1e-9
    uni = coverage(gt, uniform_highlights(T, 0.145), T, rule="any_overlap")
    assert coverage(gt, hl, T, rule="any_overlap") >= uni - 1e-12
