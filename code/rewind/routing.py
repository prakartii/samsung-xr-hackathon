"""Dataset routing from video content only (ARCHITECTURE.md §6.1).

Input is the per-keyframe CLIP probability that the frame is a cooking scene. Never
the manifest, never the filename (CLAUDE.md §6.1): FPVSum filenames contain the
category and EPIC ids have their own pattern, so either would be a hardcoded shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Dataset = Literal["epic_kitchens_100", "fpvsum", "unknown"]


@dataclass(frozen=True)
class RouteDecision:
    """Routing result.

    dataset:    "epic_kitchens_100" / "fpvsum" when confident, "unknown" when the
                margin is below `min_conf` (the uncertain route, ARCHITECTURE.md §4).
    route_conf: |p_cooking - cooking_thr|, the margin from the threshold.
    p_cooking:  mean CLIP P(cooking) over keyframes. On the uncertain route,
                `p_cooking >= cooking_thr` gives the label and highlight route.
    """

    dataset: Dataset
    route_conf: float
    p_cooking: float


def route(p_cooking_per_frame: np.ndarray, cooking_thr: float, min_conf: float) -> RouteDecision:
    """Route a video from per-keyframe P(cooking) values in [0, 1]."""
    p = np.asarray(p_cooking_per_frame, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("need a non-empty 1-D array of keyframe probabilities")
    if not np.all(np.isfinite(p)) or p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("keyframe probabilities must be finite and in [0, 1]")

    p_mean = float(p.mean())
    margin = abs(p_mean - cooking_thr)
    if margin < min_conf:
        dataset: Dataset = "unknown"
    elif p_mean >= cooking_thr:
        dataset = "epic_kitchens_100"
    else:
        dataset = "fpvsum"
    return RouteDecision(dataset=dataset, route_conf=margin, p_cooking=p_mean)
