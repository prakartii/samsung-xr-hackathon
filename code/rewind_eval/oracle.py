"""Oracle highlights from ground truth: an upper reference (ARCHITECTURE.md §9.5). Pure.

Greedy over candidate windows (each length in ``lengths``, starts on a ``stride`` grid):
repeatedly add the window with the most *new* gain per second that doesn't overlap or
touch a chosen one and fits the budget. Same rules as our output: ≥ 4 s windows,
total ≤ budget_frac · T, strictly disjoint.

- coverage oracle: gain = distinct GT (verb, noun) pairs newly captured;
- F1 oracle: gain = GT highlight frames newly covered.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np

from rewind.constants import MIN_SEG_EMIT, SHORT_VIDEO_T
from rewind.types import Segment
from rewind_eval.baselines import uniform_highlights
from rewind_eval.coverage import CaptureRule, GTAction, coverage

_GAP = 0.01  # chosen windows must be at least this far apart (no touching)


def candidate_windows(duration: float, lengths: Sequence[float], stride: float) -> list[Segment]:
    out = []
    for length in lengths:
        n = math.floor((duration - length) / stride + 1e-9) + 1
        out += [(round(i * stride, 2), round(i * stride + length, 2)) for i in range(max(0, n))]
    return out


def _greedy(
    windows: list[Segment],
    budget: float,
    gains: Callable[[], Sequence[float]],
    commit: Callable[[int], None],
) -> list[Segment]:
    """``gains()`` returns the current gain of every window; ``commit(i)`` records a pick."""
    lengths = [e - s for s, e in windows]
    alive = [True] * len(windows)
    chosen: list[Segment] = []
    used = 0.0
    while True:
        g = gains()
        best, best_score = -1, 0.0
        for i in range(len(windows)):
            if not alive[i] or g[i] <= 1e-12 or used + lengths[i] > budget + 1e-9:
                continue
            if g[i] / lengths[i] > best_score:
                best, best_score = i, g[i] / lengths[i]
        if best < 0:
            return sorted(chosen)
        b = windows[best]
        chosen.append(b)
        used += lengths[best]
        commit(best)
        for i, w in enumerate(windows):  # retire windows that now overlap or touch
            if alive[i] and w[0] < b[1] + _GAP and b[0] < w[1] + _GAP:
                alive[i] = False


def _budget(duration: float, budget_frac: float) -> float:
    # Short videos (and 27–27.93 s) still get one MIN_SEG_EMIT window, as postprocess does.
    return max(budget_frac * duration, MIN_SEG_EMIT if duration >= MIN_SEG_EMIT else duration)


def _captures(
    gt: Sequence[GTAction], windows: list[Segment], rule: CaptureRule, theta: float
) -> list[set[tuple[str, str]]]:
    """For each window, the GT pairs it captures on its own (same rule as is_captured)."""
    if not gt:
        return [set() for _ in windows]
    starts = np.array([g[0][0] for g in gt])
    ends = np.array([g[0][1] for g in gt])
    pairs = [(v, n) for _, v, n in gt]
    out = []
    for ws, we in windows:
        ov = np.clip(np.minimum(ends, we) - np.maximum(starts, ws), 0.0, None)
        if rule == "any_overlap":
            hit = ov > 1e-9
        else:
            length = ends - starts
            hit = (length > 0) & (ov >= (theta - 1e-9) * np.where(length > 0, length, 1.0))
        out.append({pairs[i] for i in np.flatnonzero(hit)})
    return out


def coverage_oracle(
    gt: Sequence[GTAction],
    duration: float,
    *,
    budget_frac: float,
    lengths: Sequence[float] = (MIN_SEG_EMIT,),
    stride: float = 0.5,
    rule: CaptureRule = "any_overlap",
    theta: float = 0.5,
) -> list[Segment]:
    if duration < SHORT_VIDEO_T:
        lengths = (min(MIN_SEG_EMIT, duration),)
    windows = candidate_windows(duration, lengths, stride)
    caps = _captures(gt, windows, rule, theta)
    keep = [i for i, c in enumerate(caps) if c]
    windows, caps = [windows[i] for i in keep], [caps[i] for i in keep]
    have: set[tuple[str, str]] = set()

    def gains() -> list[float]:
        return [float(len(c - have)) for c in caps]

    def commit(i: int) -> None:
        have.update(caps[i])

    picked = _greedy(windows, _budget(duration, budget_frac), gains, commit)
    # Greedy max coverage isn't guaranteed to beat a fixed layout (review of #14 found a
    # case). Keep whichever of greedy and uniform windows captures more, so the oracle is
    # never below the uniform baseline.
    uniform = uniform_highlights(duration, budget_frac)
    cov = lambda hl: coverage(gt, hl, duration, rule=rule, theta=theta) or 0.0  # noqa: E731
    return uniform if cov(uniform) > cov(picked) else picked


def f1_oracle(
    times: np.ndarray,
    gt_positive: np.ndarray,
    duration: float,
    *,
    budget_frac: float,
    lengths: Sequence[float] = (MIN_SEG_EMIT,),
    stride: float = 0.5,
) -> list[Segment]:
    windows = candidate_windows(duration, lengths, stride)
    times = np.asarray(times, dtype=float)
    gt = np.asarray(gt_positive, dtype=bool)
    lo = np.searchsorted(times, [w[0] for w in windows], side="left")
    hi = np.searchsorted(times, [w[1] for w in windows], side="left")  # end exclusive
    csum = np.concatenate(([0], np.cumsum(gt)))
    # Static gains: chosen windows are disjoint, so a frame inside a chosen window is
    # never inside a window still in play.
    static = (csum[hi] - csum[lo]).astype(float)
    return _greedy(windows, _budget(duration, budget_frac), lambda: static, lambda _i: None)
