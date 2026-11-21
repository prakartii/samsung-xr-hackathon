"""EPIC highlight selection: expected-coverage windows + hedge windows. Pure.

The selector spends (1 − h)·B on expected coverage (``select_windows``) and the rest
on hedges, B = budget_frac · T. ``mode="dedup"`` is the label-dedup baseline from the
kill rule (PROGRESS.md P5): each cluster collapses to its top-1 pair, everything else
the same. Output still goes through ``postprocess`` (rounding, short videos).
"""

from collections.abc import Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import Literal

import numpy as np

from rewind.config import Config
from rewind.constants import MIN_SEG_EMIT
from rewind.select.coverage import select_windows
from rewind.select.hedge import hedge_windows
from rewind.types import Cluster, Segment

Mode = Literal["expected", "dedup"]


def top1_clusters(clusters: Sequence[Cluster]) -> list[Cluster]:
    """Each cluster's distribution collapsed to its most likely pair (label dedup)."""
    out = []
    for c in clusters:
        if not c.pair_dist:
            continue
        best = min(c.pair_dist.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        out.append(replace(c, pair_dist=MappingProxyType({best: 1.0})))
    return out


def select_epic(
    clusters: Sequence[Cluster],
    duration: float,
    cfg: Config,
    *,
    mode: Mode = "expected",
    profile: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[list[Segment], list[float]]:
    """Highlight segments and a value per segment (hedges get 0)."""
    s, h = cfg.select, cfg.hedge
    budget = s.budget_frac * duration
    cl = top1_clusters(clusters) if mode == "dedup" else list(clusters)
    lengths: Sequence[float] = s.window_lengths
    share = h.share
    if budget < MIN_SEG_EMIT:
        # Short video (or T just above 27 s): the budget can't hold a window, but the kit
        # needs one. Pick the single best minimum window instead of leaving it to
        # postprocess, which would centre it on the middle of the video.
        length = min(MIN_SEG_EMIT, duration)
        budget, lengths, share = length, (length,), 0.0
    sel = select_windows(
        cl,
        duration,
        budget=(1.0 - share) * budget,
        lengths=lengths,
        stride=s.stride,
        merge_gap=cfg.post.merge_gap_epic,
        rule=s.capture_rule,
        theta=s.fraction_theta,
        tau=s.fraction_tau,
    )
    used = sum(e - a for a, e in sel.segments)
    hedges = hedge_windows(
        sel.segments,
        clusters,
        duration,
        budget=budget - used,
        gap=max(h.gap, cfg.post.merge_gap_epic),
        strong=h.strong,
        min_spacing=h.min_spacing,
        profile=profile,
    )
    segs = list(sel.segments) + hedges
    values = list(sel.values) + [0.0] * len(hedges)
    order = sorted(range(len(segs)), key=lambda i: segs[i])
    return [segs[i] for i in order], [values[i] for i in order]
