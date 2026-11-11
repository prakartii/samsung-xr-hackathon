"""Verb and noun detections -> paired (verb, noun) proposals and actions (ARCHITECTURE.md §5.1).

ActionFormer's EPIC models detect verbs and nouns separately: each detection is
(segment, class key, probability). Pairing:

1. Anchor on each verb detection. Its group = verb detections and noun detections
   overlapping it with tIoU >= `match_tiou`. An anchor without a matching noun yields
   nothing (an action needs both).
2. One proposal per group: an anchor is kept only if it is its group's top verb
   (so k verb labels on the same span give one proposal, not k duplicates).
3. Top-k verbs and nouns (max probability per key), then every (v, n) of them that is
   in the kit vocabulary is scored

       score(v, n) = (p(v) * p(n) * prior(v, n) ** alpha) ** score_power

   `prior(v, n)` is the train-split co-occurrence P(n | v) with additive smoothing,
   precomputed offline (tools/build_vn_prior.py); inference never reads annotations.
   Every factor is in [0, 1], so the score is too; the power is a fixed monotone map
   applied identically to every video (no per-video min-max, which would break
   cross-video ranking for mAP).
4. Actions = every (proposal segment, v, n, score) over all proposals, sorted by score,
   capped at `max_per_video` (<= the kit's 2000).

Pure: plain data in, plain data out.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np

from rewind.constants import MAX_ACTIONS
from rewind.types import Proposal, Segment

Detection = tuple[Segment, str, float]  # (segment, class key, probability)
Action = tuple[Segment, str, str, float]  # (segment, verb, noun, score)


@dataclass(frozen=True)
class PairingParams:
    """`pairing` in configs/default.yaml."""

    match_tiou: float
    k: int
    alpha: float
    prior_eps: float
    score_power: float


@dataclass(frozen=True)
class VNPrior:
    """Row-normalized co-occurrence P(noun | verb) over kit keys."""

    verbs: tuple[str, ...]
    nouns: tuple[str, ...]
    matrix: np.ndarray  # (V, N), rows sum to 1

    @classmethod
    def from_counts(
        cls, verbs: Sequence[str], nouns: Sequence[str], counts: np.ndarray, eps: float
    ) -> VNPrior:
        """prior(v, n) = (count(v, n) + eps) / sum_n' (count(v, n') + eps)."""
        counts = np.asarray(counts, dtype=np.float64)
        if counts.shape != (len(verbs), len(nouns)):
            raise ValueError(f"counts {counts.shape} != ({len(verbs)}, {len(nouns)})")
        if eps <= 0:
            raise ValueError("prior_eps must be > 0 so unseen pairs keep a nonzero prior")
        smoothed = counts + eps
        return cls(tuple(verbs), tuple(nouns), smoothed / smoothed.sum(axis=1, keepdims=True))

    def __post_init__(self) -> None:
        object.__setattr__(self, "_vi", {v: i for i, v in enumerate(self.verbs)})
        object.__setattr__(self, "_ni", {n: i for i, n in enumerate(self.nouns)})

    def get(self, verb: str, noun: str) -> float:
        return float(self.matrix[self._vi[verb], self._ni[noun]])  # type: ignore[attr-defined]


def tiou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Temporal IoU between every row of a (A, 2) and b (B, 2) -> (A, B)."""
    inter = np.clip(
        np.minimum(a[:, None, 1], b[None, :, 1]) - np.maximum(a[:, None, 0], b[None, :, 0]),
        0.0,
        None,
    )
    union = (a[:, 1] - a[:, 0])[:, None] + (b[:, 1] - b[:, 0])[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _top_k(dets: Sequence[Detection], idx: np.ndarray, k: int) -> tuple[tuple[str, float], ...]:
    """Max probability per key among dets[idx], top k by (prob desc, key asc)."""
    best: dict[str, float] = {}
    for i in idx:
        _, key, p = dets[i]
        best[key] = max(best.get(key, 0.0), p)
    return tuple(sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:k])


def pair_proposals(
    verb_dets: Sequence[Detection],
    noun_dets: Sequence[Detection],
    prior: VNPrior,
    verb_vocab: Collection[str],
    noun_vocab: Collection[str],
    params: PairingParams,
) -> list[Proposal]:
    """One proposal per verb group that has at least one matching noun; sorted by start."""
    if not verb_dets or not noun_dets:
        return []
    vsegs = np.array([d[0] for d in verb_dets], dtype=np.float64)
    nsegs = np.array([d[0] for d in noun_dets], dtype=np.float64)
    vv = tiou_matrix(vsegs, vsegs) >= params.match_tiou
    vn = tiou_matrix(vsegs, nsegs) >= params.match_tiou

    proposals = []
    for i, (seg, _, _) in enumerate(verb_dets):
        noun_idx = np.flatnonzero(vn[i])
        if noun_idx.size == 0:
            continue
        group = np.flatnonzero(vv[i])
        verbs = _top_k(verb_dets, group, params.k)
        leaders = [j for j in group if verb_dets[j][1:] == verbs[0]]
        if i != min(leaders):
            continue  # another detection leads this group and anchors its proposal
        nouns = _top_k(noun_dets, noun_idx, params.k)
        pairs = [
            (v, n, (pv * pn * prior.get(v, n) ** params.alpha) ** params.score_power)
            for v, pv in verbs
            if v in verb_vocab
            for n, pn in nouns
            if n in noun_vocab
        ]
        if not pairs:
            continue
        pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
        proposals.append(Proposal(seg, verbs, nouns, tuple(pairs[: params.k])))
    proposals.sort(key=lambda pr: (pr.segment[0], pr.segment[1]))
    return proposals


def proposals_to_actions(
    proposals: Sequence[Proposal], max_per_video: int, min_score: float
) -> list[Action]:
    """Flatten to actions, highest score first (ties: earlier start, then keys)."""
    if not 0 <= max_per_video <= MAX_ACTIONS:
        raise ValueError(f"max_per_video must be in [0, {MAX_ACTIONS}]")
    actions = [(pr.segment, v, n, s) for pr in proposals for v, n, s in pr.pairs if s >= min_score]
    actions.sort(key=lambda a: (-a[3], a[0][0], a[0][1], a[1], a[2]))
    return actions[:max_per_video]
