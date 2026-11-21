"""Verb and noun detections -> paired (verb, noun) proposals and actions (ARCHITECTURE.md §5.1).

ActionFormer's EPIC models detect verbs and nouns separately: each detection is
(segment, class key, probability). Pairing:

1. Greedy anchors: visit verb detections by (probability desc, start, key). A verb
   anchors a proposal if it has at least one noun detection with tIoU >= `match_tiou`
   and doesn't overlap (tIoU >= `match_tiou`) an anchor already emitted. Its group =
   the verb and noun detections overlapping it.
2. So k verb labels on the same span give one proposal, not k duplicates, and a
   confident verb with no matching noun never blocks an overlapping verb that has one
   (an earlier "top verb of its group" rule dropped such actions).
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
    top_k: int
    alpha: float
    eps: float
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
            raise ValueError("pairing.eps must be > 0 so unseen pairs keep a nonzero prior")
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


def match_lists(a: np.ndarray, b: np.ndarray, thr: float) -> list[np.ndarray]:
    """For each row of a (A, 2): sorted indices j of b (B, 2) with tIoU(a_i, b_j) >= thr.

    Same result as `tiou_matrix(a, b) >= thr` row by row, without the dense (A, B)
    matrix: for thr > 0 a match needs a real overlap, so only segments of b that start
    before a_i ends and end after it starts are scored. (thr <= 0 matches every pair,
    as the dense form does.) Zero-shot detections on a 2000 s video are ~6000 per
    side, where the dense matrices took ~1 GB.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if len(b) == 0:
        return [np.array([], dtype=np.intp) for _ in range(len(a))]
    if thr <= 0:
        return [np.arange(len(b), dtype=np.intp) for _ in range(len(a))]
    order = np.argsort(b[:, 0], kind="stable")
    bs = b[order]
    out = []
    for s, e in a:
        k = np.searchsorted(bs[:, 0], e, side="left")  # b starts strictly before a ends
        cand = order[:k][bs[:k, 1] > s]  # ...and ends strictly after a starts
        if cand.size:
            sub = tiou_matrix(np.array([[s, e]]), b[cand])[0]
            cand = np.sort(cand[sub >= thr])
        out.append(cand.astype(np.intp))
    return out


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
    keep: int | None = None,
    noun_tiou: float | None = None,
) -> list[Proposal]:
    """One proposal per verb group that has at least one matching noun; sorted by start.

    ``keep``: pairs kept per proposal (default ``params.top_k``)."""
    if not verb_dets or not noun_dets:
        return []
    vsegs = np.array([d[0] for d in verb_dets], dtype=np.float64)
    nsegs = np.array([d[0] for d in noun_dets], dtype=np.float64)
    vv = match_lists(vsegs, vsegs, params.match_tiou)  # neighbours, not dense rows
    vn = match_lists(vsegs, nsegs, params.match_tiou if noun_tiou is None else noun_tiou)

    proposals = []
    anchor_set: set[int] = set()
    order = sorted(
        range(len(verb_dets)), key=lambda i: (-verb_dets[i][2], verb_dets[i][0], verb_dets[i][1])
    )
    for i in order:
        seg = verb_dets[i][0]
        noun_idx = vn[i]
        if noun_idx.size == 0:
            continue  # no noun: can't form an action, and doesn't block its neighbours
        if anchor_set.intersection(vv[i].tolist()):
            continue  # an overlapping, more confident verb already anchors this span
        group = vv[i]
        verbs = _top_k(verb_dets, group, params.top_k)
        nouns = _top_k(noun_dets, noun_idx, params.top_k)
        pairs = [
            (v, n, (pv * pn * prior.get(v, n) ** params.alpha) ** params.score_power)
            for v, pv in verbs
            if v in verb_vocab
            for n, pn in nouns
            if n in noun_vocab
        ]
        if not pairs:
            continue  # nothing in the kit vocabulary: not an anchor either
        anchor_set.add(i)
        pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
        proposals.append(Proposal(seg, verbs, nouns, tuple(pairs[: keep or params.top_k])))
    proposals.sort(key=lambda pr: (pr.segment[0], pr.segment[1]))
    return proposals


def class_soft_nms(actions: Sequence[Action], sigma: float) -> list[Action]:
    """Gaussian soft-NMS within each (verb, noun) class: a lower-scored action is decayed
    by exp(-tIoU^2 / sigma) for every higher-scored kept action of the same class it
    overlaps. sigma <= 0 returns the actions unchanged. Output re-sorted by score."""
    if sigma <= 0:
        return list(actions)
    by_cls: dict[tuple[str, str], list[list]] = {}
    for seg, v, n, s in actions:
        by_cls.setdefault((v, n), []).append([seg, s])
    out = []
    for (v, n), items in by_cls.items():
        pool = sorted(items, key=lambda x: (-x[1], x[0]))
        kept = []
        while pool:
            best = pool.pop(0)
            kept.append(best)
            for it in pool:
                t = _tiou(best[0], it[0])
                if t > 0:
                    it[1] *= float(np.exp(-(t * t) / sigma))
            pool.sort(key=lambda x: (-x[1], x[0]))
        out += [(seg, v, n, s) for seg, s in kept]
    out.sort(key=lambda a: (-a[3], a[0][0], a[0][1], a[1], a[2]))
    return out


def _tiou(a: Segment, b: Segment) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def proposals_to_actions(
    proposals: Sequence[Proposal], max_per_video: int, min_score: float
) -> list[Action]:
    """Flatten to actions, highest score first (ties: earlier start, then keys)."""
    if not 0 <= max_per_video <= MAX_ACTIONS:
        raise ValueError(f"max_per_video must be in [0, {MAX_ACTIONS}]")
    actions = [(pr.segment, v, n, s) for pr in proposals for v, n, s in pr.pairs if s >= min_score]
    actions.sort(key=lambda a: (-a[3], a[0][0], a[0][1], a[1], a[2]))
    return actions[:max_per_video]
