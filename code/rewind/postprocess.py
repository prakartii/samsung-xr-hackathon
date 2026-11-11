"""Turn any list of candidate segments into highlights that satisfy every kit rule.

ARCHITECTURE.md §5.6. Pure: segments and numbers in, segments out. Shared by both
routes and by the fallback.

Guarantees for every output (T = video duration, B = budget_frac · T):
- sorted, strictly non-overlapping, inside [0, T], 2-decimal timestamps;
- every segment ≥ MIN_SEG (4.0 s) after rounding;
- total ≤ B, except that one MIN_SEG_EMIT window is always kept when B is smaller
  than that (it then still fits the kit's 15% when T ≥ SHORT_VIDEO_T);
- T < SHORT_VIDEO_T: exactly one segment, shaped by ``short_video_mode``;
- never empty when T ≥ MIN_SEG.
"""

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np

from rewind.constants import (
    MIN_SEG,
    MIN_SEG_EMIT,
    SCORING_BUDGET,
    SHORT_VIDEO_T,
    TIME_DECIMALS,
)
from rewind.types import Segment

_EPS = 1e-9
_SCALE = 10**TIME_DECIMALS

# (t, v): a value profile over time, e.g. activity on the dense grid. Optional.
Profile = tuple[np.ndarray, np.ndarray]


def postprocess(
    segments: Sequence[Segment],
    values: Sequence[float],
    duration: float,
    *,
    budget_frac: float,
    merge_gap: float,
    short_video_mode: Literal["min_seg", "within_budget"],
    profile: Profile | None = None,
) -> tuple[Segment, ...]:
    """Clamp, merge, drop < 4 s, trim to budget, handle short videos, round.

    values[i] is the caller's value for segments[i] (higher = keep). It decides which
    segment is trimmed first and where a single window goes. ``profile`` (optional)
    decides which end of a segment is trimmed and where a fallback window is centred.
    """
    if len(values) != len(segments):
        raise ValueError(f"{len(segments)} segments but {len(values)} values")
    T = float(duration)
    if T < MIN_SEG:
        # No valid highlight exists (R11 needs 4 s). Emit the whole video; the
        # validator will flag it and the caller's fallback handles it.
        return ((0.0, _floor(T)),) if T > 0 else ()

    items = _clamp(segments, values, T)
    if T < SHORT_VIDEO_T:
        return (_single_window(items, T, short_video_mode, profile),)

    items = _merge(items, merge_gap)
    # Drop anything shorter than MIN_SEG_EMIT, not just MIN_SEG: a 4.01 s segment could
    # round to 3.99 s. MIN_SEG_EMIT survives inward rounding (≥ 4.03 s).
    items = [(s, e, v) for s, e, v in items if e - s >= MIN_SEG_EMIT - _EPS]
    kept = _trim_to_budget(items, budget_frac * T, profile)
    if not kept:
        # Nothing fits the budget (or nothing came in): one MIN_SEG_EMIT window at the
        # best candidate seen before trimming, else at the profile peak or the middle.
        c = _best_center(items, T, profile)
        kept = [(*_place(c, MIN_SEG_EMIT, T), 0.0)]
    return tuple(_round_inside(s, e) for s, e, _ in sorted(kept))


def _clamp(
    segments: Sequence[Segment], values: Sequence[float], T: float
) -> list[tuple[float, float, float]]:
    out = []
    for (s, e), v in zip(segments, values, strict=True):
        if not (math.isfinite(s) and math.isfinite(e) and math.isfinite(v)):
            continue
        s, e = max(0.0, float(s)), min(T, float(e))
        if e - s > _EPS:
            out.append((s, e, float(v)))
    return sorted(out)


def _merge(items: list[tuple[float, float, float]], gap: float) -> list[tuple[float, float, float]]:
    """Merge segments that overlap, touch, or are closer than ``gap``. Value = max.

    Gaps under one rounding step (0.01 s) always merge: rounding could otherwise make
    the two segments touch, which R10 forbids.
    """
    merged: list[tuple[float, float, float]] = []
    for s, e, v in items:  # sorted by start
        if merged and s - merged[-1][1] < max(gap, 1 / _SCALE):
            ps, pe, pv = merged[-1]
            merged[-1] = (ps, max(pe, e), max(pv, v))
        else:
            merged.append((s, e, v))
    return merged


def _trim_to_budget(
    items: list[tuple[float, float, float]], budget: float, profile: Profile | None
) -> list[tuple[float, float, float]]:
    """Shrink the lowest-value segment (never below MIN_SEG_EMIT), else drop it."""
    items = list(items)
    while items and sum(e - s for s, e, _ in items) > budget + _EPS:
        excess = sum(e - s for s, e, _ in items) - budget
        i = min(range(len(items)), key=lambda k: (items[k][2], -items[k][0]))
        s, e, v = items[i]
        room = (e - s) - MIN_SEG_EMIT
        if room <= _EPS:
            del items[i]
            continue
        cut = min(room, excess)
        if _value_at(profile, s) <= _value_at(profile, e):
            items[i] = (s + cut, e, v)  # the start is worth less: trim it
        else:
            items[i] = (s, e - cut, v)
    return items


def _single_window(
    items: list[tuple[float, float, float]],
    T: float,
    mode: Literal["min_seg", "within_budget"],
    profile: Profile | None,
) -> Segment:
    length = min(MIN_SEG_EMIT, T) if mode == "min_seg" else SCORING_BUDGET * T
    s, e = _place(_best_center(items, T, profile), length, T)
    if mode == "within_budget":
        # Round inward so the budget still holds; R11 fails by design in this mode.
        return (_ceil(s), max(_ceil(s) + 1 / _SCALE, _floor(e)))
    return _round_inside(s, e)


def _best_center(
    items: list[tuple[float, float, float]], T: float, profile: Profile | None
) -> float:
    if items:
        s, e, _ = max(items, key=lambda it: (it[2], -it[0]))
        return (s + e) / 2
    if profile is not None and len(profile[0]):
        t, v = profile
        return float(t[int(np.argmax(v))])
    return T / 2


def _place(center: float, length: float, T: float) -> tuple[float, float]:
    """A window of ``length`` centred on ``center``, shifted to fit inside [0, T]."""
    length = min(length, T)
    s = min(max(0.0, center - length / 2), T - length)
    return (s, s + length)


def _value_at(profile: Profile | None, x: float) -> float:
    if profile is None or not len(profile[0]):
        return 0.0
    return float(np.interp(x, profile[0], profile[1]))


def _ceil(x: float) -> float:
    q = round(x * _SCALE)
    if q / _SCALE < x - _EPS:
        q += 1
    return q / _SCALE


def _floor(x: float) -> float:
    q = round(x * _SCALE)
    if q / _SCALE > x + _EPS:
        q -= 1
    return q / _SCALE


def _round_inside(s: float, e: float) -> Segment:
    """Round start up and end down so the segment only shrinks (stays in bounds)."""
    return (_ceil(s), _floor(e))
