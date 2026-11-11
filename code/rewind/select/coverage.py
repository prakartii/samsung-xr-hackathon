"""Expected-coverage window selection for EPIC videos (ARCHITECTURE.md §5.4). Pure.

State is the *union* U of selected windows, merged like post-processing merges
highlights (gaps < merge_gap filled), so the model scores exactly what is emitted.
A cluster c with pair distribution p_c is captured by U with probability
cap_c(U) (``capture_prob`` of how much of its span U covers). Treating each cluster as
one action whose pair is drawn from p_c:

    P_miss(pair | U) = Π_c (1 − p_c(pair) · cap_c(U))
    E(U)             = Σ_pair (1 − P_miss(pair | U))     expected distinct pairs captured
    gain(w | U)      = E(U + w) − E(U);   cost(w | U) = |U + w| − |U|  (new seconds)

(U + w: the union after adding window w and merging.)

Greedy by gain / cost with lazy re-evaluation (a popped entry from before the last
pick is re-scored and pushed back). A window that doesn't fit the budget when popped is
dropped. Stops when the heap is empty.
"""

import bisect
import heapq
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from rewind.select.candidates import CaptureRule, candidate_windows, capture_prob
from rewind.types import Cluster, Segment

Pair = tuple[str, str]
_EPS = 1e-12


@dataclass(frozen=True)
class Selection:
    segments: tuple[Segment, ...]  # merged union components, sorted
    values: tuple[float, ...]  # expected clusters captured per segment (for postprocess)
    expected: float  # E(U): expected number of distinct pairs captured


def merge_union(segs: list[Segment], w: Segment, merge_gap: float) -> list[Segment]:
    """Add ``w`` to sorted disjoint ``segs``, merging gaps < max(merge_gap, 0.01 s).

    Same rule as postprocess._merge, so the union is exactly what will be emitted.
    """
    gap = max(merge_gap, 0.01)
    s, e = w
    out: list[Segment] = []
    for a, b in segs:
        if s - b >= gap or a - e >= gap:
            out.append((a, b))
        else:
            s, e = min(s, a), max(e, b)
    out.append((s, e))
    return sorted(out)


def covered_seconds(segs: Sequence[Segment], span: Segment) -> float:
    return sum(max(0.0, min(b, span[1]) - max(a, span[0])) for a, b in segs)


def expected_coverage(
    clusters: Sequence[Cluster],
    segs: Sequence[Segment],
    *,
    rule: CaptureRule,
    theta: float,
    tau: float,
) -> float:
    """E(U) from scratch (reference implementation; the selector updates incrementally)."""
    p_miss: dict[Pair, float] = defaultdict(lambda: 1.0)
    for c in clusters:
        span_len = c.span[1] - c.span[0]
        cap = capture_prob(covered_seconds(segs, c.span), span_len, rule, theta, tau)
        for pair, p in c.pair_dist.items():
            p_miss[pair] *= 1.0 - p * cap
    return sum(1.0 - m for m in p_miss.values())


def select_windows(
    clusters: Sequence[Cluster],
    duration: float,
    *,
    budget: float,
    lengths: Sequence[float],
    stride: float,
    merge_gap: float,
    rule: CaptureRule,
    theta: float,
    tau: float,
) -> Selection:
    """Greedy expected-coverage selection within ``budget`` seconds of highlight."""
    spans = [c.span for c in clusters]
    windows = candidate_windows(duration, lengths, stride, spans)
    if not windows or budget <= 0:
        return Selection((), (), 0.0)
    order = sorted(range(len(clusters)), key=lambda i: spans[i][0])
    starts = [spans[i][0] for i in order]
    max_span = max((e - s for s, e in spans), default=0.0)

    union: list[Segment] = []
    total = 0.0
    cap = [0.0] * len(clusters)
    p_miss: dict[Pair, float] = defaultdict(lambda: 1.0)

    def affected(lo: float, hi: float) -> list[int]:
        """Clusters whose span overlaps [lo, hi]."""
        a = bisect.bisect_left(starts, lo - max_span)
        b = bisect.bisect_right(starts, hi)
        return [order[k] for k in range(a, b) if spans[order[k]][1] > lo]

    def evaluate(w: Segment) -> tuple[float, float, list[Segment], dict[int, float]]:
        new_union = merge_union(union, w, merge_gap)
        new_total = sum(b - a for a, b in new_union)
        comp = next(seg for seg in new_union if seg[0] <= w[0] and w[1] <= seg[1])
        new_caps = {}
        ratio: dict[Pair, float] = defaultdict(lambda: 1.0)
        for c in affected(comp[0], comp[1]):
            span = spans[c]
            nc = capture_prob(covered_seconds(new_union, span), span[1] - span[0], rule, theta, tau)
            if nc <= cap[c] + _EPS:
                continue
            new_caps[c] = nc
            for pair, p in clusters[c].pair_dist.items():
                old = 1.0 - p * cap[c]
                if old > _EPS:
                    ratio[pair] *= (1.0 - p * nc) / old
        gain = sum(p_miss[pair] * (1.0 - r) for pair, r in ratio.items())
        # Round away float noise (32.05 - 28.0 = 4.049999…) so equal choices tie
        # exactly and the earliest window wins, deterministically.
        return round(gain, 12), round(new_total - total, 9), new_union, new_caps

    version = [0] * len(windows)
    heap: list[tuple[float, int, int]] = []

    def push(i: int) -> None:
        gain, cost, _, _ = evaluate(windows[i])
        version[i] += 1
        if gain > _EPS and cost > _EPS:
            heapq.heappush(heap, (-gain / cost, i, version[i]))

    picks = 0
    fresh = [0] * len(windows)  # pick count at which window i was last evaluated
    for i in range(len(windows)):
        push(i)
    while heap:
        _, i, ver = heapq.heappop(heap)
        if ver != version[i]:
            continue
        if fresh[i] != picks:
            fresh[i] = picks
            push(i)
            continue
        _, cost, new_union, new_caps = evaluate(windows[i])
        if total + cost > budget + 1e-9:
            continue  # doesn't fit: dropped
        for c, nc in new_caps.items():
            for pair, p in clusters[c].pair_dist.items():
                old = 1.0 - p * cap[c]
                if old > _EPS:
                    p_miss[pair] *= (1.0 - p * nc) / old
                else:
                    p_miss[pair] = 0.0
            cap[c] = nc
        union, total = new_union, total + cost
        picks += 1

    values = tuple(
        sum(cap[c] for c in affected(s, e) if covered_seconds([(s, e)], spans[c]) > 0)
        for s, e in union
    )
    expected = sum(1.0 - m for m in p_miss.values())
    return Selection(tuple(union), values, expected)
