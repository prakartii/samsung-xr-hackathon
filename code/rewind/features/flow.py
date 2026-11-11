"""Optical-flow motion signals (ARCHITECTURE.md §2 `Signals.flow` / `global_motion`).

For each pair of consecutive sampled frames, Farneback flow (low resolution) is split
into:

- global motion: magnitude of the median flow vector, i.e. the camera's ego-motion
  (head turns, walking). Robust to a moving hand covering part of the frame.
- residual motion: mean magnitude of (flow - median), i.e. what moves *relative to
  the camera*: hands and manipulated objects. This is the activity signal.

Both are z-scored per video. The decomposition is pure; `flow_signals` is the OpenCV
edge that runs Farneback on decoded frames.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass

import cv2
import numpy as np

# Bump when the algorithm changes in a way the parameters don't capture
# (CLAUDE.md §6.8: change the extractor -> bump the version -> old caches ignored).
# v2 (2026-11-11): PyAV decoding by presentation time; v1 used OpenCV, which truncated
# EPIC GoPro files (tests/integration/test_decode_epic.py).
_ALGO_VERSION = "flow-v2"


@dataclass(frozen=True)
class FlowParams:
    """Farneback parameters: `features.flow` in configs/default.yaml."""

    width: int
    pyr_scale: float
    levels: int
    winsize: int
    iterations: int
    poly_n: int
    poly_sigma: float


def flow_cache_version(params: FlowParams, dense_fps: float) -> str:
    """Cache version string: algorithm version + hash of every parameter that matters."""
    blob = json.dumps({**asdict(params), "dense_fps": dense_fps}, sort_keys=True)
    return f"{_ALGO_VERSION}-{hashlib.sha1(blob.encode()).hexdigest()[:12]}"


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


def motion_series(flows: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Raw (residual, global) series for k + 1 frames from k consecutive flow fields.

    Flow i describes frames i -> i+1 and is assigned to frame i+1. Frame 0 has no
    predecessor, so it copies frame 1's value (avoids a fake dip at t = 0). No flows
    (a single frame) -> one zero.
    """
    values = [decompose_flow(f) for f in flows]
    if not values:
        return np.zeros(1), np.zeros(1)
    residual = np.array([values[0][0]] + [r for r, _ in values], dtype=np.float64)
    global_ = np.array([values[0][1]] + [g for _, g in values], dtype=np.float64)
    return residual, global_


def _farneback(prev_gray: np.ndarray, gray: np.ndarray, p: FlowParams) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        p.pyr_scale,
        p.levels,
        p.winsize,
        p.iterations,
        p.poly_n,
        p.poly_sigma,
        0,
    )


def raw_motion(
    frames: Iterable[tuple[float, np.ndarray]], p: FlowParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, residual, global) in pixels per sampled frame, before z-scoring."""
    times: list[float] = []

    def flows() -> Iterator[np.ndarray]:
        prev = None
        for t, frame in frames:
            times.append(t)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                yield _farneback(prev, gray, p)
            prev = gray

    residual, global_ = motion_series(flows())
    if not times:
        raise ValueError("no frames decoded")
    return np.asarray(times, dtype=np.float64), residual, global_


def flow_signals(
    frames: Iterable[tuple[float, np.ndarray]], p: FlowParams
) -> dict[str, np.ndarray]:
    """Z-scored residual and global motion for (t, BGR frame) pairs from `iter_frames`.

    Frames should already be resized to `p.width` (pass `width=p.width` to
    `iter_frames`). Returns {"t": (N,), "flow": (N,), "global_motion": (N,)}.
    """
    t, residual, global_ = raw_motion(frames, p)
    return {
        "t": t,
        "flow": zscore(residual),
        "global_motion": zscore(global_),
    }
