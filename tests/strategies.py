"""Shared Hypothesis strategies (CLAUDE.md §7.3 rule 8)."""

from hypothesis import strategies as st

durations = st.floats(min_value=5.0, max_value=2000.0, allow_nan=False, allow_infinity=False)


@st.composite
def segments_and_values(draw, duration: float, max_n: int = 300):
    """0–max_n candidate segments that may spill outside [0, T] or overlap freely."""
    n = draw(st.integers(min_value=0, max_value=max_n))
    starts = draw(st.lists(st.floats(-10.0, duration + 10.0), min_size=n, max_size=n))
    lengths = draw(st.lists(st.floats(0.0, 60.0), min_size=n, max_size=n))
    values = draw(st.lists(st.floats(0.0, 1.0), min_size=n, max_size=n))
    return [(s, s + L) for s, L in zip(starts, lengths, strict=True)], values
