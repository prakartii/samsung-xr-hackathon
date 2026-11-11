"""Proposals → clusters of one distinct action each (ARCHITECTURE.md §5.2). Pure.

Two proposals are linked when they overlap enough in time (tIoU ≥ ``tiou``) and
either agree on their top-1 (verb, noun) pair or look alike (cosine similarity of
their CLIP embeddings ≥ ``sim``, when embeddings are given). Connected components are
clusters. A cluster's ``pair_dist`` is the score-weighted mix of its members' pair
distributions, cut to the top ``k``, renormalized and temperature-scaled.
"""

from collections import defaultdict
from collections.abc import Sequence
from types import MappingProxyType

import numpy as np

from rewind.types import Cluster, Proposal, Segment

Pair = tuple[str, str]


def tiou(a: Segment, b: Segment) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _top1(p: Proposal) -> Pair | None:
    return (p.pairs[0][0], p.pairs[0][1]) if p.pairs else None


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def cluster_proposals(
    proposals: Sequence[Proposal],
    *,
    tiou_thr: float,
    sim_thr: float,
    k: int,
    temperature: float,
    embeddings: np.ndarray | None = None,
) -> tuple[Cluster, ...]:
    """Cluster proposals; clusters come out sorted by span start.

    embeddings: optional (n, D) L2-normalized mean CLIP embedding per proposal.
    """
    n = len(proposals)
    order = sorted(range(n), key=lambda i: proposals[i].segment)
    parent = list(range(n))
    for a_pos, i in enumerate(order):
        si = proposals[i].segment
        for j in order[a_pos + 1 :]:
            sj = proposals[j].segment
            if sj[0] >= si[1]:
                break  # sorted by start: nothing later overlaps i
            if tiou(si, sj) < tiou_thr:
                continue
            same = _top1(proposals[i]) is not None and _top1(proposals[i]) == _top1(proposals[j])
            alike = embeddings is not None and float(embeddings[i] @ embeddings[j]) >= sim_thr
            if same or alike:
                parent[_find(parent, i)] = _find(parent, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in order:
        groups[_find(parent, i)].append(i)
    members = sorted(groups.values(), key=lambda m: (proposals[m[0]].segment, m))
    return tuple(
        _make_cluster(cid, [proposals[i] for i in m], tuple(sorted(m)), k, temperature)
        for cid, m in enumerate(members)
    )


def _make_cluster(
    cid: int, members: list[Proposal], idx: tuple[int, ...], k: int, temperature: float
) -> Cluster:
    mass: dict[Pair, float] = defaultdict(float)
    confidence = 0.0
    for p in members:
        total = sum(s for _, _, s in p.pairs)
        if total <= 0:
            continue
        weight = max(s for _, _, s in p.pairs)  # the member's confidence
        confidence = max(confidence, weight)
        for v, n, s in p.pairs:
            mass[(v, n)] += weight * s / total
    top = sorted(mass.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    dist = calibrate({pair: m for pair, m in top}, temperature)
    span = (min(p.segment[0] for p in members), max(p.segment[1] for p in members))
    return Cluster(
        cluster_id=cid,
        span=span,
        member_idx=idx,
        pair_dist=MappingProxyType(dist),
        confidence=confidence,
    )


def calibrate(mass: dict[Pair, float], temperature: float) -> dict[Pair, float]:
    """p ∝ m^(1/T), normalized to sum 1 (T = 1 → plain normalization)."""
    if not mass:
        return {}
    powered = {pair: m ** (1.0 / temperature) for pair, m in mass.items()}
    z = sum(powered.values())
    return {pair: v / z for pair, v in powered.items()} if z > 0 else {}
