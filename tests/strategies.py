"""Shared Hypothesis strategies (CLAUDE.md §7.3.8): durations 5-2000 s, signals."""

import numpy as np
from hypothesis import strategies as st

durations = st.floats(min_value=5.0, max_value=2000.0, allow_nan=False, allow_infinity=False)


@st.composite
def dense_signal(draw, duration: float, rate: float = 4.0):
    """(t, values) on the dense grid with random finite values in [-3, 3]."""
    n = max(1, int(duration * rate))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    values = np.random.default_rng(seed).uniform(-3.0, 3.0, n)
    return np.arange(n) / rate, values
