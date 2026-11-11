"""Simulated detector proposals from ground truth, for selector work before P4. Pure.

Each GT action is dropped with probability ``miss``; kept ones get Gaussian boundary
jitter and a top-k pair list where the true pair is ranked first with probability
``top1_acc`` (else somewhere in 2..k, or missing), with distractors that share the
verb or the noun. False positives arrive at ``fp_per_min`` with random pairs.
Seeded: the same inputs always give the same proposals.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass

from rewind.types import Proposal
from rewind_eval.coverage import GTAction

Pair = tuple[str, str]


@dataclass(frozen=True)
class Noise:
    name: str
    miss: float
    jitter: float
    top1_acc: float
    topk: int
    fp_per_min: float


CLEAN = Noise("clean", miss=0.0, jitter=0.0, top1_acc=1.0, topk=1, fp_per_min=0.0)
# Rough stand-in for an ActionFormer-class detector on EPIC-100 (see PROGRESS.md P5).
NOISY = Noise("noisy", miss=0.3, jitter=1.0, top1_acc=0.4, topk=5, fp_per_min=1.0)


def _distractors(true: Pair, pool: Sequence[Pair], rng: random.Random, n: int) -> list[Pair]:
    near = [p for p in pool if p != true and (p[0] == true[0] or p[1] == true[1])]
    rest = [p for p in pool if p != true and p not in near]
    out: list[Pair] = []
    for src in (near, rest):
        cand = [p for p in src if p not in out]
        rng.shuffle(cand)
        out += cand[: n - len(out)]
        if len(out) >= n:
            break
    return out


def _pairs(true: Pair | None, pool: Sequence[Pair], noise: Noise, rng: random.Random):
    k = noise.topk
    if true is None:
        ranked = _distractors(("", ""), pool, rng, k)
    else:
        ranked = _distractors(true, pool, rng, k)
        u = rng.random()
        if u < noise.top1_acc:
            ranked = [true, *ranked][:k]
        elif k > 1 and u < noise.top1_acc + (1 - noise.top1_acc) / 2:
            pos = rng.randint(1, k - 1)
            ranked = ([*ranked[:pos], true, *ranked[pos:]])[:k]
    conf = rng.uniform(0.3, 0.9)
    weights = [0.5**r for r in range(len(ranked))]
    z = sum(weights)
    return tuple((v, n, conf * w / z) for (v, n), w in zip(ranked, weights, strict=True))


def simulate(
    gt: Sequence[GTAction], duration: float, noise: Noise, pool: Sequence[Pair], seed: int
) -> list[Proposal]:
    rng = random.Random(seed)
    out = []
    for (s, e), v, n in gt:
        if rng.random() < noise.miss:
            continue
        js = s + rng.gauss(0.0, noise.jitter) if noise.jitter else s
        je = e + rng.gauss(0.0, noise.jitter) if noise.jitter else e
        lo, hi = sorted((js, je))
        js = min(max(0.0, lo), duration - 0.2)
        je = min(duration, max(hi, js + 0.2))
        exact = noise.topk == 1 and noise.top1_acc >= 1.0
        pairs = ((v, n, 1.0),) if exact else _pairs((v, n), pool, noise, rng)
        out.append(Proposal((js, je), (), (), pairs))
    n_fp = round(noise.fp_per_min * duration / 60.0)
    for _ in range(n_fp):
        s = rng.uniform(0.0, max(0.0, duration - 1.0))
        out.append(
            Proposal(
                (s, min(duration, s + rng.uniform(1.0, 5.0))),
                (),
                (),
                _pairs(None, pool, noise, rng),
            )
        )
    return out
