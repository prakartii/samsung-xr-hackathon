"""FPVSum importance proxy and highlight segments (ARCHITECTURE.md §6.2).

Per dense timestep:

    imp(t) = w_flow * z(flow) + w_hands * hands + w_global * z(global_motion)
             + w_scene * z(scene_change)

then Gaussian smoothing with sigma seconds. Scene change = 1 - cosine similarity of
consecutive 1 fps CLIP embeddings, interpolated to the dense grid. (`flow` and
`global_motion` arrive z-scored from features/flow.py.)

Segments: shots start where scene change exceeds `shot_thr`. Inside each shot,
windows of `min_len`..`max_len` s are scored by mean imp; the best non-overlapping
windows are taken greedily until the budget is used. The result still goes through
post-processing (§5.6), which owns merging, short videos and rounding.

Pure: arrays in, segments out. Weights are tuned on FPVSum CV folds only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rewind.constants import MIN_SEG_EMIT
from rewind.features.clip import l2_normalize
from rewind.features.flow import zscore
from rewind.types import Segment


@dataclass(frozen=True)
class ImportanceParams:
    """`importance` in configs/default.yaml."""

    w_flow: float
    w_hands: float
    w_global: float
    w_scene: float
    sigma: float
    shot_thr: float
    min_len: float
    max_len: float
    len_step: float
    stride: float

    def __post_init__(self) -> None:
        if self.min_len < MIN_SEG_EMIT:
            raise ValueError(f"importance.min_len must be >= {MIN_SEG_EMIT} (kit 4 s rule)")
        if self.max_len < self.min_len or self.len_step <= 0 or self.stride <= 0:
            raise ValueError("need max_len >= min_len and positive len_step / stride")


def clip_change(emb: np.ndarray) -> np.ndarray:
    """(M,) 1 - cos(emb[j-1], emb[j]) at j; j = 0 copies j = 1 (0 for a single frame)."""
    e = l2_normalize(np.asarray(emb, dtype=np.float64))
    if len(e) < 2:
        return np.zeros(len(e))
    d = 1.0 - np.sum(e[1:] * e[:-1], axis=1)
    return np.concatenate([[d[0]], d])


def gaussian_smooth(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Gaussian filter with reflected edges; sigma 0 returns x unchanged."""
    x = np.asarray(x, dtype=np.float64)
    if sigma_samples <= 0 or len(x) < 2:
        return x.copy()
    radius = max(1, int(np.ceil(3 * sigma_samples)))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_samples) ** 2)
    k /= k.sum()
    padded = np.pad(x, radius, mode="reflect" if len(x) > radius else "edge")
    return np.convolve(padded, k, mode="valid")


def importance_signal(
    t_dense: np.ndarray,
    flow: np.ndarray,
    hands: np.ndarray,
    global_motion: np.ndarray,
    t_clip: np.ndarray,
    clip_emb: np.ndarray,
    p: ImportanceParams,
) -> np.ndarray:
    """(N,) smoothed importance on the dense grid."""
    scene = np.interp(t_dense, t_clip, clip_change(clip_emb))
    raw = (
        p.w_flow * np.asarray(flow)
        + p.w_hands * np.asarray(hands)
        + p.w_global * np.asarray(global_motion)
        + p.w_scene * zscore(scene)
    )
    rate = 1.0 / np.median(np.diff(t_dense)) if len(t_dense) > 1 else 1.0
    return gaussian_smooth(raw, p.sigma * rate)


def shots(t_clip: np.ndarray, clip_emb: np.ndarray, duration: float, thr: float) -> list[Segment]:
    """Contiguous shots covering [0, duration]; a cut where clip change > thr."""
    change = clip_change(clip_emb)
    cuts = [float(t_clip[j]) for j in range(1, len(t_clip)) if change[j] > thr]
    bounds = [0.0, *[c for c in cuts if 0.0 < c < duration], float(duration)]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a]


def _candidates(span: Segment, p: ImportanceParams) -> list[Segment]:
    s0, e0 = span
    out = []
    for length in np.arange(p.min_len, p.max_len + 1e-9, p.len_step):
        for start in np.arange(s0, e0 - length + 1e-9, p.stride):
            out.append((float(start), float(start + length)))
    return out


def select_segments(
    t_dense: np.ndarray,
    imp: np.ndarray,
    shot_spans: list[Segment],
    duration: float,
    budget_sec: float,
    p: ImportanceParams,
) -> list[Segment]:
    """Greedy best-mean-importance windows inside shots, non-overlapping, within budget.

    If no shot is long enough for a min_len window, windows may span shots. If the
    budget is below min_len (short video), returns the single best min_len window
    (post-processing applies the short-video rule).
    """
    t = np.asarray(t_dense)
    imp = np.asarray(imp)
    cands = [c for span in shot_spans for c in _candidates(span, p)]
    if not cands:
        cands = _candidates((0.0, float(duration)), p)
    if not cands:  # video shorter than min_len: post-processing decides
        return []

    def score(seg: Segment) -> float:
        inside = (t >= seg[0]) & (t < seg[1])
        return float(imp[inside].mean()) if inside.any() else -np.inf

    ranked = sorted(cands, key=lambda c: (-score(c), c[0], c[1] - c[0]))
    if budget_sec < p.min_len:
        return [ranked[0]]
    chosen: list[Segment] = []
    used = 0.0
    for s, e in ranked:
        if used + (e - s) > budget_sec + 1e-9:
            continue
        if any(s < ce and cs < e for cs, ce in chosen):
            continue
        chosen.append((s, e))
        used += e - s
        if budget_sec - used < p.min_len:
            break
    return sorted(chosen)
