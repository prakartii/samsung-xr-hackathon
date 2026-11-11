"""Video label (ARCHITECTURE.md §6.3, §4 step 4).

- EPIC route -> ("cooking", 1.0).
- FPVSum route -> CLIP zero-shot over the 14 FPVSum categories (kit keys, so `ski`,
  not `skiing`): softmax per keyframe over prompt-ensembled class embeddings, averaged
  over `n_keyframes` evenly spaced 1 fps embeddings; score = that mean probability of
  the argmax class.
- Uncertain route -> `cooking` if mean CLIP P(cooking) >= `cooking_thr`, else the
  FPVSum zero-shot label, replaced by `other` when its score < `other_thr`.

Pure: embeddings and the routing decision in, (name, score) out.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rewind.features.clip import zero_shot_probs
from rewind.routing import RouteDecision

COOKING = "cooking"
OTHER = "other"


def keyframe_indices(m: int, n: int) -> np.ndarray:
    """n evenly spaced indices into m frames (all of them if m <= n)."""
    if m <= 0:
        raise ValueError("no frames")
    if m <= n:
        return np.arange(m)
    return np.round(np.linspace(0, m - 1, n)).astype(int)


def fpvsum_label(
    clip_emb: np.ndarray,
    class_emb: np.ndarray,
    keys: Sequence[str],
    logit_scale: float,
    n_keyframes: int,
) -> tuple[str, float]:
    """Zero-shot (name, score) over FPVSum classes (`keys` excludes cooking and other)."""
    frames = np.asarray(clip_emb)[keyframe_indices(len(clip_emb), n_keyframes)]
    probs = zero_shot_probs(frames, class_emb, logit_scale).mean(axis=0)
    best = int(np.argmax(probs))
    return keys[best], float(probs[best])


def choose_label(
    route: RouteDecision, cooking_thr: float, fpvsum: tuple[str, float], other_thr: float
) -> tuple[str, float]:
    """Final label for a routed video; `fpvsum` is the zero-shot FPVSum label."""
    if route.dataset == "epic_kitchens_100":
        return COOKING, 1.0
    if route.dataset == "fpvsum":
        return fpvsum
    if route.p_cooking >= cooking_thr:  # uncertain route, leaning cooking
        return COOKING, float(route.p_cooking)
    name, score = fpvsum
    return (OTHER, score) if score < other_thr else (name, score)
