"""Per-step classifier evidence for clusters (C1). Pure.

ActionFormer's released EPIC features come with per-step verb and noun class
probabilities (`v_prob` (K, 97), `n_prob` (K, 300); one step per 16/30 s, the same grid
as the features). They are a second opinion, independent of the detector's own
segment scores, on *which* action happens inside a cluster's span and *whether* one does.

`with_step_evidence` scales each pair p of a cluster's distribution by
`e(p) ** power`, where e(p) = sqrt(mean v_prob[v] * mean n_prob[n]) over the steps whose
centre lies in the cluster span (the nearest step if none does). Power 0 = unchanged.
No renormalization: a cluster whose pairs the classifier doesn't support is worth less.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from types import MappingProxyType

import numpy as np

from rewind.types import Cluster


def span_steps(t: np.ndarray, span: tuple[float, float]) -> np.ndarray:
    """Indices of steps with centre in [start, end]; the nearest step if none."""
    idx = np.flatnonzero((t >= span[0]) & (t <= span[1]))
    if idx.size == 0:
        mid = 0.5 * (span[0] + span[1])
        idx = np.array([int(np.argmin(np.abs(t - mid)))])
    return idx


def with_step_evidence(
    clusters: Sequence[Cluster],
    t: np.ndarray,
    v_prob: np.ndarray,
    n_prob: np.ndarray,
    verb_keys: Sequence[str],
    noun_keys: Sequence[str],
    power: float,
) -> list[Cluster]:
    """Clusters with pair weights scaled by per-step classifier evidence (see module)."""
    if power <= 0 or len(t) == 0:
        return list(clusters)
    vi = {k: i for i, k in enumerate(verb_keys) if i < v_prob.shape[1]}
    ni = {k: i for i, k in enumerate(noun_keys) if i < n_prob.shape[1]}
    out = []
    for c in clusters:
        steps = span_steps(t, c.span)
        mv, mn = v_prob[steps].mean(axis=0), n_prob[steps].mean(axis=0)
        dist = {}
        for (v, n), p in c.pair_dist.items():
            if v in vi and n in ni:
                e = float(np.sqrt(max(mv[vi[v]], 0.0) * max(mn[ni[n]], 0.0)))
            else:
                e = 0.0  # a class the classifier has no output for: no support
            dist[(v, n)] = p * min(1.0, e) ** power
        out.append(replace(c, pair_dist=MappingProxyType(dist)))
    return out
