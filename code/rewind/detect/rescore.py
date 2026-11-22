"""Learned output-action score (Campaign #4). Pure.

Action mAP ranks each (verb, noun) class's actions across all videos by score, so the
score should estimate P(this action is a true positive). The pairing score
(p_v * p_n) ** power ignores evidence that is visible at inference: the verb and noun
ranks inside their proposal, how well a noun detection of that class lines up with the
verb segment, whether a higher-scored action of the same class already covers the span
(a duplicate is a false positive), and how many actions of that class the video has.

A logistic model over those features (no GT, no video identity) is fit offline on
EPIC-100 train videos (tools/train_rescore.py) and stored in configs/default.yaml
(`actions.rescore_coef`, `rescore_mean`, `rescore_std`); an empty coefficient list = off.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from rewind.detect.pairing import Action, Detection, match_lists, tiou_matrix
from rewind.types import Proposal

FEATURES = (
    "score",
    "verb_prob",
    "noun_prob",
    "log_verb_rank",
    "log_noun_rank",
    "log_pair_rank",
    "vn_tiou",
    "log_duration",
    "position",
    "log_overlap",
    "log_class_rank",
    "log_class_count",
    "log_verb_count",
    "log_noun_count",
    "score_ratio",
    "dup_tiou",
)


def action_features(
    proposals: Sequence[Proposal], noun_dets: Sequence[Detection], min_score: float
) -> tuple[list[Action], np.ndarray]:
    """Every (proposal segment, verb, noun, pairing score) with score >= ``min_score``,
    sorted like `proposals_to_actions` (score desc, start, end, keys), and their
    (n, len(FEATURES)) feature matrix in the same order.

    Per action: pairing score; the verb's and noun's probability; log(1 + 0-based rank)
    of the verb, noun and pair within the proposal; best tIoU between the segment and a
    noun detection of that class; log duration; midpoint / last proposal end;
    log(1 + other proposals overlapping the segment); log(1 + rank among same-class
    actions of the video); log(1 + count) of same-class, same-verb and same-noun actions;
    score / the video's top score; best tIoU with a higher-ranked same-class action.
    """
    rows: list[tuple[int, str, str, float, int, int, int]] = []
    for i, pr in enumerate(proposals):
        vr = {v: r for r, (v, _) in enumerate(pr.verbs)}
        nr = {n: r for r, (n, _) in enumerate(pr.nouns)}
        for r, (v, n, s) in enumerate(pr.pairs):
            if s >= min_score:
                rows.append((i, v, n, s, vr.get(v, len(vr)), nr.get(n, len(nr)), r))
    rows.sort(key=lambda a: (-a[3], *proposals[a[0]].segment, a[1], a[2]))
    actions: list[Action] = [(proposals[i].segment, v, n, s) for i, v, n, s, *_ in rows]
    if not rows:
        return actions, np.zeros((0, len(FEATURES)))

    segs = np.array([p.segment for p in proposals], dtype=np.float64)
    overlap = [len(m) - 1 for m in match_lists(segs, segs, 1e-9)]  # minus itself
    extent = float(segs[:, 1].max())
    by_noun: dict[str, np.ndarray] = {}
    for seg, key, _ in noun_dets:
        by_noun.setdefault(key, []).append(seg)  # type: ignore[arg-type]
    by_noun = {k: np.array(v, dtype=np.float64) for k, v in by_noun.items()}
    vn_cache: dict[tuple[int, str], float] = {}
    top = rows[0][3]
    cls_count: dict[tuple[str, str], int] = {}
    verb_count: dict[str, int] = {}
    noun_count: dict[str, int] = {}
    for _, v, n, *_ in rows:
        cls_count[(v, n)] = cls_count.get((v, n), 0) + 1
        verb_count[v] = verb_count.get(v, 0) + 1
        noun_count[n] = noun_count.get(n, 0) + 1
    seen: dict[tuple[str, str], list] = {}
    out = []
    for i, v, n, s, vr, nr, r in rows:
        pr = proposals[i]
        a, b = pr.segment
        if (i, n) not in vn_cache:
            nd = by_noun.get(n)
            best = float(tiou_matrix(segs[i : i + 1], nd).max()) if nd is not None else 0.0
            vn_cache[(i, n)] = best
        prev = seen.setdefault((v, n), [])
        dup = float(tiou_matrix(segs[i : i + 1], np.array(prev)).max()) if prev else 0.0
        out.append([
            s,
            dict(pr.verbs).get(v, 0.0),
            dict(pr.nouns).get(n, 0.0),
            math.log1p(vr),
            math.log1p(nr),
            math.log1p(r),
            vn_cache[(i, n)],
            math.log(max(b - a, 1e-2)),
            0.5 * (a + b) / extent if extent > 0 else 0.0,
            math.log1p(overlap[i]),
            math.log1p(len(prev)),
            math.log1p(cls_count[(v, n)]),
            math.log1p(verb_count[v]),
            math.log1p(noun_count[n]),
            s / top if top > 0 else 0.0,
            dup,
        ])  # fmt: skip
        prev.append(pr.segment)
    return actions, np.array(out, dtype=float)


def rescore_probs(
    x: np.ndarray, coef: Sequence[float], mean: Sequence[float], std: Sequence[float]
) -> np.ndarray:
    """P(true positive) per row: sigmoid(b + w . (x - mean) / std); coef = [b, w...]."""
    if x.shape[0] == 0:
        return np.zeros(0)
    sd = np.where(np.asarray(std) > 0, np.asarray(std), 1.0)
    logit = coef[0] + ((x - np.asarray(mean)) / sd) @ np.asarray(coef[1:])
    return 1.0 / (1.0 + np.exp(-logit))


def rescore_actions(
    proposals: Sequence[Proposal],
    noun_dets: Sequence[Detection],
    min_score: float,
    coef: Sequence[float],
    mean: Sequence[float],
    std: Sequence[float],
    old_power: float,
    max_per_video: int,
) -> list[Action]:
    """Actions scored P(TP) * old_score ** old_power, highest first, capped."""
    actions, x = action_features(proposals, noun_dets, min_score)
    p = rescore_probs(x, coef, mean, std)
    out = [
        (seg, v, n, float(q * s**old_power)) for (seg, v, n, s), q in zip(actions, p, strict=True)
    ]
    out.sort(key=lambda a: (-a[3], a[0][0], a[0][1], a[1], a[2]))
    return out[:max_per_video]
