"""Action coverage (ARCHITECTURE.md §9.1): share of distinct GT (verb, noun) pairs
captured by highlights ≥ 4 s, times an over-budget penalty. Pure.

Capture rules (the official one is unknown, PROGRESS.md Q4):
- ``any_overlap``: a GT instance is captured if a valid highlight overlaps it at all.
- ``fraction``: captured if the valid highlights cover at least ``theta`` of its span.
"""

from collections.abc import Iterable, Sequence
from typing import Literal

from rewind.constants import MIN_SEG, SCORING_BUDGET
from rewind.types import Segment

CaptureRule = Literal["any_overlap", "fraction"]
GTAction = tuple[Segment, str, str]  # (segment, verb, noun)
_EPS = 1e-9


def overlap(a: Segment, b: Segment) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def is_captured(
    span: Segment, highlights: Sequence[Segment], rule: CaptureRule, theta: float
) -> bool:
    """Whether one GT span is captured. ``highlights`` must be valid and disjoint."""
    if rule == "any_overlap":
        return any(overlap(span, h) > _EPS for h in highlights)
    length = span[1] - span[0]
    if length <= 0:
        return False
    return sum(overlap(span, h) for h in highlights) / length >= theta - _EPS


def budget_penalty(highlights: Iterable[Segment], duration: float, lam: float) -> float:
    """1 within 15% of T; falls linearly by ``lam`` per extra 15% over (floored at 0)."""
    total = sum(e - s for s, e in highlights)
    over = max(0.0, total / duration - SCORING_BUDGET)
    return max(0.0, 1.0 - lam * over / SCORING_BUDGET)


def coverage(
    gt: Sequence[GTAction],
    highlights: Sequence[Segment],
    duration: float,
    *,
    rule: CaptureRule,
    theta: float = 0.5,
    lam: float = 1.0,
) -> float | None:
    """Coverage for one video, or None if it has no GT actions (excluded from means).

    Highlights under 4 s are ignored for capture (kit §4) but still count toward the
    budget, since they are still part of the submitted total.
    """
    pairs = {(v, n) for _, v, n in gt}
    if not pairs:
        return None
    valid = [h for h in highlights if h[1] - h[0] >= MIN_SEG - _EPS]
    captured = {(v, n) for span, v, n in gt if is_captured(span, valid, rule, theta)}
    return len(captured) / len(pairs) * budget_penalty(highlights, duration, lam)


def mean_ignoring_none(values: Iterable[float | None]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0
