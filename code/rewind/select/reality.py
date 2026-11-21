"""Learned cluster-reality score (Campaign #2). Pure.

A logistic model estimating whether a cluster contains a real action, from features
available at inference only (no GT, no video identity). Weights are learned offline on
EPIC-100 train videos (tools/train_reality.py) and stored in configs/default.yaml
(`select.reality_coef`, `reality_mean`, `reality_std`); an empty coefficient list = off.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from rewind.types import Cluster

FEATURES = (
    "confidence",
    "log_members",
    "log_duration",
    "entropy",
    "top1",
    "margin",
    "position",
    "log_density",
    "conf_rank",
)


def cluster_features(clusters: Sequence[Cluster], duration: float) -> np.ndarray:
    """(n, len(FEATURES)) feature matrix, one row per cluster, in input order."""
    n = len(clusters)
    if n == 0:
        return np.zeros((0, len(FEATURES)))
    conf = np.array([c.confidence for c in clusters], dtype=float)
    mids = np.array([0.5 * (c.span[0] + c.span[1]) for c in clusters])
    order = conf.argsort(kind="stable")
    rank = np.empty(n)
    rank[order] = np.arange(n) / max(n - 1, 1)
    rows = []
    for i, c in enumerate(clusters):
        p = np.array(sorted(c.pair_dist.values(), reverse=True), dtype=float)
        p = p / p.sum() if p.sum() > 0 else np.ones(1)
        ent = float(-(p * np.log(np.clip(p, 1e-12, None))).sum() / math.log(max(len(p), 2)))
        top1 = float(p[0])
        margin = float(p[0] - (p[1] if len(p) > 1 else 0.0))
        dens = int(np.sum(np.abs(mids - mids[i]) <= 15.0)) - 1
        rows.append([
            conf[i],
            math.log1p(len(c.member_idx)),
            math.log(max(c.span[1] - c.span[0], 1e-2)),
            ent,
            top1,
            margin,
            mids[i] / duration if duration > 0 else 0.0,
            math.log1p(dens),
            rank[i],
        ])  # fmt: skip
    return np.array(rows, dtype=float)


def reality_scores(
    clusters: Sequence[Cluster],
    duration: float,
    coef: Sequence[float],
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """P(real) per cluster: sigmoid(b + w . (x - mean) / std); coef = [b, w...]."""
    x = cluster_features(clusters, duration)
    if x.shape[0] == 0:
        return np.zeros(0)
    z = (x - np.asarray(mean)) / np.where(np.asarray(std) > 0, np.asarray(std), 1.0)
    logit = coef[0] + z @ np.asarray(coef[1:])
    return 1.0 / (1.0 + np.exp(-logit))


PAIR_FEATURES = (*FEATURES, "pair_prob", "log_pair_rank", "pair_ratio")


def pair_features(clusters: Sequence[Cluster], duration: float) -> tuple[np.ndarray, list]:
    """(rows, keys): one row per (cluster, pair) = the cluster's features + the pair's
    probability within the cluster, log(1 + rank) and probability / top-1 probability."""
    cx = cluster_features(clusters, duration)
    rows, keys = [], []
    for i, c in enumerate(clusters):
        items = sorted(c.pair_dist.items(), key=lambda kv: (-kv[1], kv[0]))
        total = sum(p for _, p in items) or 1.0
        top = items[0][1] / total if items else 1.0
        for r, (pair, p) in enumerate(items):
            q = p / total
            rows.append([*cx[i], q, math.log1p(r), q / top if top > 0 else 0.0])
            keys.append((i, pair))
    return (np.array(rows, dtype=float) if rows else np.zeros((0, len(PAIR_FEATURES)))), keys


def pair_scores(
    clusters: Sequence[Cluster],
    duration: float,
    coef: Sequence[float],
    mean: Sequence[float],
    std: Sequence[float],
) -> list[dict]:
    """Per cluster, {pair: P(this pair really happens in the cluster span)}."""
    x, keys = pair_features(clusters, duration)
    out: list[dict] = [{} for _ in clusters]
    if x.shape[0] == 0:
        return out
    sd = np.where(np.asarray(std) > 0, np.asarray(std), 1.0)
    p = 1.0 / (1.0 + np.exp(-(coef[0] + ((x - np.asarray(mean)) / sd) @ np.asarray(coef[1:]))))
    for (i, pair), v in zip(keys, p, strict=True):
        out[i][pair] = float(v)
    return out
