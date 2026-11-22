"""Learned per-timestep highlight importance (FPVSum route). Pure.

A logistic model on the dense grid that replaces the hand-weighted
``importance.importance_signal`` when ``learned_importance.coef`` is non-empty. Features
are available at inference only: the motion signals (residual flow, hands, camera
motion, CLIP scene change) with their temporal context, position in the video, and the
cosine similarity of each 1 fps CLIP frame to a few generic text prompts
(``learned_importance.prompts``). Weights are fit offline on YouTube Highlights labels
(tools/train_importance.py); the output is the logit, Gaussian-smoothed with
``learned_importance.sigma`` seconds, and goes to ``importance.select_segments``
unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from rewind.features.flow import zscore
from rewind.select.importance import clip_change, gaussian_smooth

BASE = ("flow", "hands", "gm", "scene", "hands_z")
CONTEXT = ("", "_s1", "_s3", "_s6", "_s12", "_lmax3", "_d2", "_dog")
POSITION = ("pos", "edge", "pos2")


def feature_names(n_prompts: int) -> tuple[str, ...]:
    """Column names of ``step_features`` in order."""
    motion = tuple(b + c for b in BASE for c in CONTEXT)
    return (*motion, *POSITION, *(f"txt{j}" for j in range(n_prompts)))


def local_max(x: np.ndarray, half: int) -> np.ndarray:
    """Max over [i - half, i + half], edges padded with the edge value."""
    x = np.asarray(x, dtype=np.float64)
    if half <= 0 or len(x) == 0:
        return x.copy()
    return sliding_window_view(np.pad(x, half, mode="edge"), 2 * half + 1).max(axis=1)


def step_features(
    t_dense: np.ndarray,
    flow: np.ndarray,
    hands: np.ndarray,
    global_motion: np.ndarray,
    t_clip: np.ndarray,
    clip_emb: np.ndarray,
    prompt_sims: np.ndarray,
    duration: float,
) -> np.ndarray:
    """(N, len(feature_names(P))) features on the dense grid.

    prompt_sims: (M, P) cosine similarity of each 1 fps CLIP frame to each prompt.
    Per base signal x: x, Gaussian-smoothed at 1/3/6/12 s, local max (+-3 s) of the
    3 s-smoothed signal, gradient of the 2 s-smoothed signal, 3 s minus 12 s smoothing.
    Each prompt column: per-video z-score interpolated to the dense grid, smoothed 3 s.
    """
    t = np.asarray(t_dense, dtype=np.float64)
    rate = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    tc = np.asarray(t_clip, dtype=np.float64)
    h = np.asarray(hands, dtype=np.float64)
    base = (
        zscore(flow),
        h,
        zscore(global_motion),
        zscore(np.interp(t, tc, clip_change(clip_emb))),
        zscore(h),
    )
    cols = []
    for x in base:
        s3 = gaussian_smooth(x, 3 * rate)
        cols += [
            x,
            gaussian_smooth(x, 1 * rate),
            s3,
            gaussian_smooth(x, 6 * rate),
            gaussian_smooth(x, 12 * rate),
            local_max(s3, int(3 * rate)),
            np.gradient(gaussian_smooth(x, 2 * rate)) if len(x) > 1 else np.zeros(len(x)),
            s3 - gaussian_smooth(x, 12 * rate),
        ]
    pos = t / duration if duration > 0 else np.zeros(len(t))
    cols += [pos, np.minimum(pos, 1 - pos), pos**2]
    sims = np.asarray(prompt_sims, dtype=np.float64).reshape(len(tc), -1)
    for j in range(sims.shape[1]):
        cols.append(gaussian_smooth(zscore(np.interp(t, tc, sims[:, j])), 3 * rate))
    return np.stack(cols, axis=1) if cols else np.zeros((len(t), 0))


def learned_importance(
    features: np.ndarray,
    coef: Sequence[float],
    mean: Sequence[float],
    std: Sequence[float],
    sigma_samples: float,
) -> np.ndarray:
    """(N,) smoothed logit b + w . (x - mean) / std; coef = [b, w...]."""
    sd = np.asarray(std, dtype=np.float64)
    z = (np.asarray(features) - np.asarray(mean)) / np.where(sd > 0, sd, 1.0)
    logit = coef[0] + z @ np.asarray(coef[1:], dtype=np.float64)
    return gaussian_smooth(logit, sigma_samples)
