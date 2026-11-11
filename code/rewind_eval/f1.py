"""Frame-level highlight F1 for FPVSum (ARCHITECTURE.md §9.3). Pure.

GT positives: the top ``top_frac`` of sampled frames by mean human importance, exactly
ceil(top_frac · N) frames (ties broken by earlier time). Predicted positives: frames
inside any highlight ≥ 4 s.
"""

import math
from collections.abc import Sequence

import numpy as np

from rewind.constants import MIN_SEG, SCORING_BUDGET
from rewind.types import Segment

_EPS = 1e-9


def top_frames(importance: np.ndarray, top_frac: float = SCORING_BUDGET) -> np.ndarray:
    """Boolean mask of the top ``top_frac`` frames (exact count, stable on ties)."""
    n = len(importance)
    k = math.ceil(top_frac * n - _EPS)
    order = np.argsort(-np.asarray(importance, dtype=float), kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def highlight_mask(times: np.ndarray, highlights: Sequence[Segment]) -> np.ndarray:
    """Frames at ``times`` inside a highlight of length ≥ MIN_SEG (end exclusive)."""
    mask = np.zeros(len(times), dtype=bool)
    for s, e in highlights:
        if e - s >= MIN_SEG - _EPS:
            mask |= (times >= s) & (times < e)
    return mask


def frame_f1(
    times: np.ndarray,
    importance: np.ndarray,
    highlights: Sequence[Segment],
    top_frac: float = SCORING_BUDGET,
) -> float:
    gt = top_frames(importance, top_frac)
    pred = highlight_mask(np.asarray(times, dtype=float), highlights)
    tp = int(np.sum(gt & pred))
    if tp == 0:
        return 0.0
    precision = tp / int(np.sum(pred))
    recall = tp / int(np.sum(gt))
    return 2 * precision * recall / (precision + recall)
