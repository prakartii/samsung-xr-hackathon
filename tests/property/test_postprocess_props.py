"""Invariants of postprocess and the fallback for any duration and any input."""

from itertools import pairwise

from hypothesis import given, settings
from hypothesis import strategies as st

from rewind.constants import MIN_SEG, MIN_SEG_EMIT, SCORING_BUDGET, SHORT_VIDEO_T
from rewind.fallback import fallback_result
from rewind.output import to_document
from rewind.postprocess import postprocess
from rewind.schema import validate
from rewind.vocab import load_vocab
from tests.strategies import durations, segments_and_values

BUDGET = 0.145
EPS = 1e-9


@st.composite
def cases(draw):
    T = draw(durations)
    segs, vals = draw(segments_and_values(T))
    mode = draw(st.sampled_from(["min_seg", "within_budget"]))
    return T, segs, vals, mode


@settings(max_examples=300, deadline=None)
@given(cases())
def test_postprocess_output_always_satisfies_highlight_rules(case):
    T, segs, vals, mode = case

    out = postprocess(segs, vals, T, budget_frac=BUDGET, merge_gap=0.5, short_video_mode=mode)

    assert out, "never empty for T >= 4 s"
    for s, e in out:
        assert 0.0 <= s < e <= T + EPS
        assert round(s, 2) == s and round(e, 2) == e  # 2 decimals
    for (_, e0), (s1, _) in pairwise(out):
        assert e0 < s1  # sorted and strictly non-overlapping
    total = sum(e - s for s, e in out)
    if T < SHORT_VIDEO_T:
        assert len(out) == 1
        if mode == "min_seg":
            assert out[0][1] - out[0][0] >= MIN_SEG - EPS
        else:
            assert total <= SCORING_BUDGET * T + EPS
    else:
        assert all(e - s >= MIN_SEG - EPS for s, e in out)
        assert total <= max(BUDGET * T, MIN_SEG_EMIT) + EPS
        assert total <= SCORING_BUDGET * T + EPS


@settings(max_examples=200, deadline=None)
@given(durations)
def test_fallback_output_passes_strict_validation(T):
    result = fallback_result(
        "T07", "vid", T, budget_frac=BUDGET, merge_gap=0.5, short_video_mode="min_seg"
    )

    errors, _ = validate(to_document(result), load_vocab(), filename_stem="vid", strict=True)

    assert errors == []
