"""The oracle is an upper reference: never below uniform windows, always valid."""

import itertools

from hypothesis import example, given, settings
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


# Found in review: greedy alone loses to uniform windows here (0.75 vs 1.0).
COUNTEREXAMPLE = (
    62.37,
    [
        ((15.67, 18.10), "v1", "x"),
        ((17.52, 18.37), "v6", "x"),
        ((45.70, 45.90), "v3", "x"),
        ((40.45, 49.15), "v2", "x"),
        ((32.28, 32.42), "v1", "x"),
        ((29.08, 29.20), "v2", "x"),
    ],
)


@settings(max_examples=150, deadline=None)
@example(COUNTEREXAMPLE)
@given(video())
def test_coverage_oracle_is_valid_and_not_below_uniform(case):
    T, gt = case

    hl = coverage_oracle(gt, T, budget_frac=0.145)

    assert all(e - s >= 4.0 for s, e in hl)
    assert all(e0 < s1 for (_, e0), (s1, _) in itertools.pairwise(hl))
    assert sum(e - s for s, e in hl) <= 0.145 * T + 1e-9
    uni = coverage(gt, uniform_highlights(T, 0.145), T, rule="any_overlap")
    assert coverage(gt, hl, T, rule="any_overlap") >= uni - 1e-12
