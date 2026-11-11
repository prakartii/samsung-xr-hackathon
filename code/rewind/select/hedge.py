"""Hedge windows: spend a reserved slice of the budget where the detector may have
missed actions (ARCHITECTURE.md §5.5). Pure.

Centres lie on a 0.5 s grid. A centre's score is the activity profile there (motion +
hands, when available) or, without a profile, its distance to the nearest selected
highlight, so hedges spread into the largest uncovered stretches. Excluded: windows
within ``gap`` of a selected highlight or overlapping a confident cluster (the
detector already knows about those). Highest score first, centres ≥ ``min_spacing``
apart, MIN_SEG_EMIT windows, until ``budget`` seconds are used.
"""

from collections.abc import Sequence

import numpy as np

from rewind.constants import MIN_SEG_EMIT
from rewind.types import Cluster, Segment

_STEP = 0.5


def hedge_windows(
    selected: Sequence[Segment],
    clusters: Sequence[Cluster],
    duration: float,
    *,
    budget: float,
    gap: float,
    strong: float,
    min_spacing: float,
    profile: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[Segment]:
    n = int(budget // MIN_SEG_EMIT + 1e-9)
    if n <= 0 or duration < MIN_SEG_EMIT:
        return []
    half = MIN_SEG_EMIT / 2
    centres = np.arange(half, duration - half + 1e-9, _STEP)
    if len(centres) == 0:
        return []
    blocked = [(s - gap, e + gap) for s, e in selected]
    blocked += [c.span for c in clusters if c.confidence >= strong]
    ok = np.ones(len(centres), dtype=bool)
    for a, b in blocked:
        ok &= ~((centres - half < b) & (centres + half > a))
    if profile is not None and len(profile[0]):
        score = np.interp(centres, profile[0], profile[1])
    elif selected:
        mids = np.array([(s + e) / 2 for s, e in selected])
        score = np.min(np.abs(centres[:, None] - mids[None, :]), axis=1)
    else:
        score = np.zeros(len(centres))

    picked: list[float] = []
    for i in np.argsort(-score, kind="stable"):
        if len(picked) == n:
            break
        c = float(centres[i])
        if not ok[i] or any(abs(c - p) < max(min_spacing, MIN_SEG_EMIT + gap) for p in picked):
            continue
        picked.append(c)
    return sorted((round(c - half, 6), round(c + half, 6)) for c in picked)
