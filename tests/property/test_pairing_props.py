"""match_lists must equal the dense tIoU matrix thresholded, row by row."""

import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st

from rewind.detect.pairing import match_lists, tiou_matrix


@st.composite
def segments(draw, max_n=40):
    n = draw(st.integers(min_value=0, max_value=max_n))
    starts = draw(st.lists(st.floats(0.0, 100.0), min_size=n, max_size=n))
    lengths = draw(st.lists(st.floats(0.0, 20.0), min_size=n, max_size=n))  # 0-length allowed
    return np.array([[s, s + w] for s, w in zip(starts, lengths, strict=True)]).reshape(-1, 2)


@settings(max_examples=200, deadline=None)
@given(a=segments(), b=segments(), thr=st.sampled_from([0.0, 0.1, 0.3, 0.5, 0.9, 1.0]))
@example(a=np.array([[0.0, 1.0]]), b=np.array([[1.0, 2.0]]), thr=0.0)  # touching, thr 0
@example(a=np.array([[0.0, 1.0]]), b=np.array([[0.0, 1.0]]), thr=1.0)  # identical, thr 1
def test_match_lists_equal_dense_threshold(a, b, thr):
    got = match_lists(a, b, thr)
    want = (
        [np.flatnonzero(row) for row in (tiou_matrix(a, b) >= thr)]
        if len(a) and len(b)
        else [np.array([], dtype=int) for _ in range(len(a))]
    )
    assert len(got) == len(a)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(g, w)
