"""Candidate windows and the capture model (ARCHITECTURE.md §5.3). Pure."""

import math
from collections.abc import Sequence
from typing import Literal

from rewind.types import Segment

CaptureRule = Literal["any_overlap", "fraction"]
_EPS = 1e-9


def candidate_windows(
    duration: float, lengths: Sequence[float], stride: float, spans: Sequence[Segment]
) -> list[Segment]:
    """Windows of each length on a ``stride`` grid, inside [0, T], overlapping ≥ 1 span."""
    out: set[Segment] = set()
    for length in lengths:
        if length > duration + _EPS:
            continue
        last = math.floor((duration - length) / stride + _EPS)
        for s, e in spans:
            # Starts with start < e and start + length > s (positive overlap).
            lo = max(0, math.floor((s - length) / stride) + 1)
            hi = min(last, math.ceil(e / stride) - 1)
            for i in range(lo, hi + 1):
                start = round(i * stride, 6)
                if start < e - _EPS and start + length > s + _EPS:
                    out.add((start, round(start + length, 6)))
    return sorted(out)


def capture_prob(
    covered: float, span_len: float, rule: CaptureRule, theta: float, tau: float
) -> float:
    """Probability a cluster counts as captured, given ``covered`` seconds of its span."""
    if covered <= _EPS:
        return 0.0
    if rule == "any_overlap":
        return 1.0
    frac = covered / span_len if span_len > 0 else 1.0
    return 1.0 / (1.0 + math.exp(-(frac - theta) / tau))
