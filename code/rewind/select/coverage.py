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
pick is re-scored and pushed back). Windows next to each pick are re-scored at once,
because their cost can drop (overlap) and their gain rise (filled gaps), which plain
lazy evaluation misses. A window that doesn't fit the budget when popped is dropped
(a later refresh can bring it back). Stops when the heap is empty.
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
    return _merge_with(segs, w, merge_gap)[0]


def _merge_with(
    segs: list[Segment], w: Segment, merge_gap: float
) -> tuple[list[Segment], int, float]:
    """merge_union plus the merged component's index and the seconds it adds."""
    gap = max(merge_gap, 0.01)
    s, e = w
    out: list[Segment] = []
    absorbed = 0.0
    idx = -1
    for a, b in segs:
        if s - b >= gap:
            out.append((a, b))
        elif a - e >= gap:
            if idx < 0:
                idx = len(out)
                out.append((s, e))
            out.append((a, b))
        else:
            s, e = min(s, a), max(e, b)
            absorbed += b - a
    if idx < 0:
        idx = len(out)
        out.append((s, e))
    else:
        out[idx] = (s, e)
    return out, idx, (e - s) - absorbed


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
    max_len = max(lengths)
    gap_r = max(merge_gap, 0.01)
    win_starts = [w[0] for w in windows]

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
        new_union, k, added = _merge_with(union, w, merge_gap)
        comp = new_union[k]
        # Only components within one span length of the changed one can touch an
        # affected cluster; summing over all of U made long videos 4x slower.
        k0, k1 = k, k + 1
        while k0 > 0 and new_union[k0 - 1][1] > comp[0] - max_span:
            k0 -= 1
        while k1 < len(new_union) and new_union[k1][0] < comp[1] + max_span:
            k1 += 1
        near = new_union[k0:k1]
        new_caps = {}
        ratio: dict[Pair, float] = defaultdict(lambda: 1.0)
        for c in affected(comp[0], comp[1]):
            span = spans[c]
            nc = capture_prob(covered_seconds(near, span), span[1] - span[0], rule, theta, tau)
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
        return round(gain, 12), round(added, 9), new_union, new_caps

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
        # Windows next to the changed component can now be cheaper (they overlap it) or
        # gain more (filled gaps), so their stale keys may be too LOW for lazy greedy.
        # Re-score them now (review of #15: without this, a worse window was picked).
        comp = next(seg for seg in union if seg[0] <= windows[i][0] and windows[i][1] <= seg[1])
        lo = bisect.bisect_left(win_starts, comp[0] - max_len - gap_r)
        hi = bisect.bisect_right(win_starts, comp[1] + gap_r)
        for j in range(lo, hi):
            fresh[j] = picks
            push(j)

    values = tuple(
        sum(cap[c] for c in affected(s, e) if covered_seconds([(s, e)], spans[c]) > 0)
        for s, e in union
    )
    expected = sum(1.0 - m for m in p_miss.values())
    return Selection(tuple(union), values, expected)
