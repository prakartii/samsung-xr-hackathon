"""Sliding-window CLIP zero-shot detector (ARCHITECTURE.md §3.1, §4; plan §7).

Used when ActionFormer can't run: the P4 kill rule, and the uncertain route when no
SlowFast features exist (a misrouted EPIC video must never lose its actions).

Windows of `window_sec` every `stride_sec` over the 1 fps CLIP embeddings: each
window's embedding is the normalized mean of the frames inside it, scored by
zero-shot softmax against verb prompts and, separately, noun prompts. The top-k
classes per window become verb and noun detections in ActionFormer's format, so
`pairing.pair_proposals` turns them into actions unchanged.

Pure: embeddings in, detections out. Text embeddings come from the CLIP edge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rewind.detect.pairing import Detection
from rewind.features.clip import l2_normalize, zero_shot_probs


@dataclass(frozen=True)
class ZeroShotParams:
    """`zeroshot` in configs/default.yaml."""

    window_sec: float
    stride_sec: float
    top_k: int
    verb_template: str
    noun_template: str


def class_phrase(key: str) -> str:
    """Readable prompt text for a kit key: hyphens to spaces, and EPIC's
    "noun:modifier[:modifier]" order reversed ("board:chopping" -> "chopping board")."""
    parts = key.split(":")
    return " ".join(reversed(parts)).replace("-", " ")


def windows(duration: float, window_sec: float, stride_sec: float) -> list[tuple[float, float]]:
    """[s, s + window] for s = 0, stride, ... while the window fits, plus one final window
    ending exactly at T when the grid stops short of it (otherwise up to one stride at the
    end would never be covered); one clipped window if the video is shorter than a window."""
    if window_sec <= 0 or stride_sec <= 0:
        raise ValueError("window and stride must be positive")
    if duration <= window_sec:
        return [(0.0, float(duration))]
    n = int(np.floor((duration - window_sec) / stride_sec + 1e-9)) + 1
    out = [(i * stride_sec, i * stride_sec + window_sec) for i in range(n)]
    if out[-1][1] < duration - 1e-9:
        out.append((float(duration) - window_sec, float(duration)))
    return out


def window_embeddings(
    t: np.ndarray, emb: np.ndarray, spans: Sequence[tuple[float, float]]
) -> np.ndarray:
    """(W, D) normalized mean of the embeddings with s <= t < e; nearest frame if none."""
    out = np.empty((len(spans), emb.shape[1]), dtype=np.float64)
    for i, (s, e) in enumerate(spans):
        inside = (t >= s) & (t < e)
        if not inside.any():
            inside = np.zeros_like(inside)
            inside[int(np.argmin(np.abs(t - (s + e) / 2)))] = True
        out[i] = emb[inside].mean(axis=0)
    return l2_normalize(out)


def detect_zeroshot(
    t: np.ndarray,
    emb: np.ndarray,
    duration: float,
    verb_keys: Sequence[str],
    verb_text: np.ndarray,
    noun_keys: Sequence[str],
    noun_text: np.ndarray,
    logit_scale: float,
    params: ZeroShotParams,
) -> tuple[list[Detection], list[Detection]]:
    """(verb detections, noun detections), each sorted by score, highest first."""
    if len(t) == 0:
        return [], []
    spans = windows(duration, params.window_sec, params.stride_sec)
    w_emb = window_embeddings(np.asarray(t), np.asarray(emb), spans)

    def dets(keys: Sequence[str], text: np.ndarray) -> list[Detection]:
        probs = zero_shot_probs(w_emb, text, logit_scale)
        k = min(params.top_k, len(keys))
        out = []
        for span, row in zip(spans, probs, strict=True):
            for c in np.argsort(-row, kind="stable")[:k]:
                out.append((span, keys[int(c)], float(row[c])))
        out.sort(key=lambda d: (-d[2], d[0][0], d[1]))
        return out

    return dets(verb_keys, verb_text), dets(noun_keys, noun_text)
