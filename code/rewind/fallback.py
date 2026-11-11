"""The safe output: written whenever any pipeline stage fails (CLAUDE.md §6.7).

Uniform MIN_SEG_EMIT windows spread evenly over the video, up to the budget; no
actions; label "other". Pure, and valid for every duration ≥ MIN_SEG.
"""

import math
from typing import Literal

from rewind.constants import MIN_SEG_EMIT
from rewind.postprocess import postprocess
from rewind.types import Segment, VideoResult

FALLBACK_LABEL: tuple[str, float] = ("other", 0.0)


def uniform_windows(duration: float, budget_frac: float) -> tuple[Segment, ...]:
    """k = floor(budget / MIN_SEG_EMIT) windows (at least 1), centred at (i + ½)·T/k."""
    k = max(1, math.floor(budget_frac * duration / MIN_SEG_EMIT))
    step = duration / k
    half = MIN_SEG_EMIT / 2
    return tuple(((i + 0.5) * step - half, (i + 0.5) * step + half) for i in range(k))


def fallback_result(
    team_id: str,
    video_id: str,
    duration: float,
    *,
    budget_frac: float,
    merge_gap: float,
    short_video_mode: Literal["min_seg", "within_budget"],
) -> VideoResult:
    windows = uniform_windows(duration, budget_frac)
    highlights = postprocess(
        windows,
        [0.0] * len(windows),
        duration,
        budget_frac=budget_frac,
        merge_gap=merge_gap,
        short_video_mode=short_video_mode,
    )
    return VideoResult(
        team_id=team_id,
        video_id=video_id,
        duration_sec=duration,
        label=FALLBACK_LABEL,
        actions=(),
        highlights=highlights,
    )
