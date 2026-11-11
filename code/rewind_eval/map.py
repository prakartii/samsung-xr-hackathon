"""Action mAP over (verb, noun) classes (ARCHITECTURE.md §9.2). Pure.

ActivityNet-style: for each tIoU threshold and each class present in the GT, rank that
class's predictions over all videos by score (ties keep input order), greedily match
each to the unmatched GT instance of the same class in the same video with the highest
tIoU ≥ threshold, and take the area under the interpolated precision–recall curve.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from rewind.types import Segment

TIOU_THRESHOLDS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)

Pred = tuple[Segment, str, str, float]  # (segment, verb, noun, score)
GT = tuple[Segment, str, str]  # (segment, verb, noun)


def tiou(a: Segment, b: Segment) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def interpolated_ap(tp: Sequence[bool], n_gt: int) -> float:
    """Area under the PR curve with precision made monotone (all-point interpolation)."""
    if n_gt == 0:
        return 0.0
    if len(tp) == 0:
        return 0.0
    tp_arr = np.asarray(tp, dtype=float)
    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(1.0 - tp_arr)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)
    # Envelope: precision at recall r = max precision at any recall >= r.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def class_ap(
    preds: Sequence[tuple[str, Segment, float]],
    gts: Mapping[str, Sequence[Segment]],
    threshold: float,
) -> float:
    """AP for one class. preds: (video_id, segment, score); gts: video_id → spans."""
    n_gt = sum(len(v) for v in gts.values())
    order = sorted(range(len(preds)), key=lambda i: -preds[i][2])  # stable on ties
    used = {vid: [False] * len(spans) for vid, spans in gts.items()}
    tp = []
    for i in order:
        vid, seg, _ = preds[i]
        best, best_j = -1.0, -1
        for j, g in enumerate(gts.get(vid, ())):
            if used[vid][j]:
                continue
            t = tiou(seg, g)
            if t >= threshold and t > best:
                best, best_j = t, j
        if best_j >= 0:
            used[vid][best_j] = True
            tp.append(True)
        else:
            tp.append(False)
    return interpolated_ap(tp, n_gt)


def mean_ap(
    preds: Mapping[str, Sequence[Pred]],
    gts: Mapping[str, Sequence[GT]],
    thresholds: Sequence[float] = TIOU_THRESHOLDS,
) -> dict[str, float]:
    """mAP per threshold and their mean ("avg"), over classes present in the GT.

    preds/gts: video_id → list. Predictions for classes absent from the GT are
    ignored (they can't be scored per class); videos only in ``preds`` are ignored.
    """
    gt_by_class: dict[tuple[str, str], dict[str, list[Segment]]] = defaultdict(dict)
    for vid, items in gts.items():
        for seg, v, n in items:
            gt_by_class[(v, n)].setdefault(vid, []).append(seg)
    pred_by_class: dict[tuple[str, str], list[tuple[str, Segment, float]]] = defaultdict(list)
    for vid in sorted(preds):
        if vid not in gts:
            continue
        for seg, v, n, score in preds[vid]:
            pred_by_class[(v, n)].append((vid, seg, score))

    out: dict[str, float] = {}
    classes = sorted(gt_by_class)
    for thr in thresholds:
        aps = [class_ap(pred_by_class.get(c, []), gt_by_class[c], thr) for c in classes]
        out[f"{thr:.1f}"] = float(np.mean(aps)) if aps else 0.0
    out["avg"] = float(np.mean([out[f"{t:.1f}"] for t in thresholds]))
    return out
