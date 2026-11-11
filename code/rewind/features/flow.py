"""Optical-flow motion signals (ARCHITECTURE.md §2 `Signals.flow` / `global_motion`).

For each pair of consecutive sampled frames, Farneback flow (low resolution) is split
into:

- global motion: magnitude of the median flow vector, i.e. the camera's ego-motion
  (head turns, walking). Robust to a moving hand covering part of the frame.
- residual motion: mean magnitude of (flow - median), i.e. what moves *relative to
  the camera*: hands and manipulated objects. This is the activity signal.

Both are z-scored per video. The decomposition is pure.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def decompose_flow(flow: np.ndarray) -> tuple[float, float]:
    """Split one (H, W, 2) flow field into (residual magnitude, global magnitude)."""
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"flow must be (H, W, 2), got {flow.shape}")
    vectors = flow.reshape(-1, 2).astype(np.float64)
    global_vec = np.median(vectors, axis=0)
    residual = np.linalg.norm(vectors - global_vec, axis=1).mean()
    return float(residual), float(np.linalg.norm(global_vec))


def zscore(x: np.ndarray) -> np.ndarray:
    """Per-video z-score (population std). A constant signal maps to all zeros."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    std = x.std()
    if std == 0.0:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def motion_series(flows: Iterable[np.ndarray], n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw (residual, global) series of length `n_frames` from n_frames - 1 flow fields.

    Flow i describes frames i -> i+1 and is assigned to frame i+1. Frame 0 has no
    predecessor, so it copies frame 1's value (avoids a fake dip at t = 0). One frame
    → zeros.
    """
    residual = np.zeros(n_frames, dtype=np.float64)
    global_ = np.zeros(n_frames, dtype=np.float64)
    count = 0
    for i, flow in enumerate(flows, start=1):
        if i >= n_frames:
            raise ValueError(f"got more than n_frames - 1 = {n_frames - 1} flow fields")
        residual[i], global_[i] = decompose_flow(flow)
        count += 1
    if n_frames > 1:
        if count != n_frames - 1:
            raise ValueError(f"expected {n_frames - 1} flow fields, got {count}")
        residual[0], global_[0] = residual[1], global_[1]
    return residual, global_
