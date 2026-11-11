"""Pure numpy port of ActionFormer's C++ 1D NMS extension (`csrc/nms_cpu.cpp`).

REWIND local addition (see VENDORED.md): the organizers re-run `run.sh` in an
isolated environment that may have no C++ compiler, so `nms.py` falls back to this
module when `nms_1d_cpu` is not built.

Semantics match the C++ code exactly, in float32:
- `nms`: greedy suppression in descending score order (torch sort order), IoU with the
  same +1e-6 length term; returns kept indices in that order.
- `softnms`: the same selection order (first maximum wins), the same score decay per
  method (0 hard, 1 linear, 2 gaussian), and the same "swap the dropped segment with
  the last one" compaction, which decides the order of later ties.
Each pass is vectorized over the remaining segments, so cost is O(n^2) in numpy, not
in Python.
"""

import numpy as np


def _iou(x1, x2, areas, ix1, ix2, iarea):
    inter = np.maximum(np.float32(0), np.minimum(ix2, x2) - np.maximum(ix1, x1))
    return inter / (iarea + areas - inter)


def nms_np(segs: np.ndarray, order: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Hard NMS. `order` = indices sorted by descending score (as torch sorted them)."""
    n = segs.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    x1 = segs[:, 0].astype(np.float32)
    x2 = segs[:, 1].astype(np.float32)
    areas = x2 - x1 + np.float32(1e-6)
    thr = np.float32(iou_threshold)
    select = np.ones(n, dtype=bool)
    ox1, ox2, oarea = x1[order], x2[order], areas[order]
    for k in range(n):
        if not select[k]:
            continue
        ovr = _iou(ox1[k + 1 :], ox2[k + 1 :], oarea[k + 1 :], ox1[k], ox2[k], oarea[k])
        select[k + 1 :] &= ~(ovr >= thr)
    return order[select].astype(np.int64)


def softnms_np(
    segs: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    sigma: float,
    min_score: float,
    method: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft-NMS. Returns (kept original indices, dets (n_kept, 3) = x1, x2, score)."""
    n = segs.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
    x1 = segs[:, 0].astype(np.float32).copy()
    x2 = segs[:, 1].astype(np.float32).copy()
    sc = scores.astype(np.float32).copy()
    areas = x2 - x1 + np.float32(1e-6)
    inds = np.arange(n, dtype=np.int64)
    dets = np.zeros((n, 3), dtype=np.float32)
    thr, sig, mins = np.float32(iou_threshold), np.float32(sigma), np.float32(min_score)
    arrays = (x1, x2, sc, areas, inds)

    i = 0
    while i < n:
        # first maximum in [i, n) (C++ keeps the first on ties: strict <)
        m = i + int(np.argmax(sc[i:n]))
        for a in arrays:
            a[i], a[m] = a[m], a[i]
        dets[i] = (x1[i], x2[i], sc[i])

        # decay every remaining score at once
        rest = slice(i + 1, n)
        ovr = _iou(x1[rest], x2[rest], areas[rest], x1[i], x2[i], areas[i])
        if method == 0:
            weight = np.where(ovr >= thr, np.float32(0), np.float32(1))
        elif method == 1:
            weight = np.where(ovr >= thr, np.float32(1) - ovr, np.float32(1))
        else:
            weight = np.exp(-(ovr * ovr) / sig).astype(np.float32)
        sc[rest] = sc[rest] * weight.astype(np.float32)

        # C++ compaction: scanning pos upward, a dropped slot is refilled with the
        # current last segment, which is checked again at the same slot. Each
        # segment's decayed score doesn't depend on where it sits, so the drop
        # decisions are known up front; replay only the moves.
        dropped = sc[i + 1 : n] < mins
        if dropped.any():
            drop = np.zeros(n, dtype=bool)
            drop[i + 1 : n] = dropped
            end = n  # current nsegs
            for p in np.flatnonzero(drop):
                if p >= end:
                    break
                # pull from the tail, discarding dropped tail segments
                while end - 1 > p and drop[end - 1]:
                    end -= 1
                if end - 1 == p:
                    end -= 1
                    break
                last = end - 1
                for a in arrays:
                    a[p] = a[last]
                drop[p] = False
                end -= 1
            n = end
        i += 1

    return inds[:n].copy(), dets[:n].copy()


# --- drop-in replacements for the compiled extension's functions ---------------------


def nms(segs, scores, iou_threshold):
    """Same signature and return type as `nms_1d_cpu.nms` (torch tensors)."""
    import torch

    order = scores.sort(0, descending=True)[1].numpy()
    return torch.from_numpy(nms_np(segs.numpy(), order, iou_threshold))


def softnms(segs, scores, dets, iou_threshold, sigma, min_score, method):
    """Same signature as `nms_1d_cpu.softnms`: fills `dets` in place, returns indices."""
    import torch

    inds, out = softnms_np(segs.numpy(), scores.numpy(), iou_threshold, sigma, min_score, method)
    dets[: len(inds)] = torch.from_numpy(out)
    return torch.from_numpy(inds)
