"""Reference highlight sets: uniform and seeded random windows. Pure.

Both use MIN_SEG_EMIT (4.05 s) windows up to budget_frac · T, rounded inward to 2
decimals, so they obey the same rules as our system's output.
"""

import math
import random

from rewind.constants import MIN_SEG_EMIT, SHORT_VIDEO_T
from rewind.types import Segment

_SCALE = 100


def _inward(s: float, e: float) -> Segment:
    return (math.ceil(s * _SCALE - 1e-9) / _SCALE, math.floor(e * _SCALE + 1e-9) / _SCALE)


def window_count(duration: float, budget_frac: float) -> int:
    """How many 4.05 s windows fit the budget (always ≥ 1: R11 needs one)."""
    if duration < SHORT_VIDEO_T:
        return 1
    return max(1, math.floor(budget_frac * duration / MIN_SEG_EMIT))


def uniform_highlights(duration: float, budget_frac: float) -> list[Segment]:
    """k windows centred at (i + ½)·T/k."""
    k = window_count(duration, budget_frac)
    half = min(MIN_SEG_EMIT, duration) / 2
    centres = [(i + 0.5) * duration / k for i in range(k)]
    return [_inward(max(0.0, c - half), min(duration, c + half)) for c in centres]


def random_highlights(duration: float, budget_frac: float, seed: int) -> list[Segment]:
    """k non-overlapping windows at random starts (seeded; deterministic)."""
    k = window_count(duration, budget_frac)
    length = min(MIN_SEG_EMIT, duration)
    rng = random.Random(seed)
    # Choose k starts among slots of one window + 0.05 s gap so windows can't touch.
    slot = length + 0.05
    n_slots = max(1, math.floor((duration - length) / slot) + 1)
    picks = sorted(rng.sample(range(n_slots), min(k, n_slots)))
    return [_inward(i * slot, i * slot + length) for i in picks]
