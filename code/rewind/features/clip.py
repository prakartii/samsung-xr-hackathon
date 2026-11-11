"""CLIP image embeddings and zero-shot scoring (ARCHITECTURE.md §2 `Signals.clip_emb`).

Used by routing (P(cooking), §6.1), clustering similarity (§5.2), scene change for
the FPVSum importance proxy (§6.2) and the FPVSum label (§6.3).

The math is pure: L2 normalization, prompt ensembles (mean of a class's template
embeddings), and zero-shot softmax. `OpenClipEncoder` is the model edge; tests use a
fake `Encoder`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from rewind.video_io import resize_to_width

# Bump when the algorithm changes in a way the parameters don't capture (CLAUDE.md §6.8).
# v2 (2026-11-11): PyAV decoding by presentation time; v1 used OpenCV, which truncated
# EPIC GoPro files (tests/integration/test_decode_epic.py).
_ALGO_VERSION = "clip-v2"


@dataclass(frozen=True)
class ClipParams:
    """`features.clip` in configs/default.yaml."""

    model: str
    weights: str
    width: int
    batch_size: int


class Encoder(Protocol):
    """What the pipeline needs from a CLIP model. Outputs are L2-normalized, float32."""

    logit_scale: float

    def encode_images(self, rgb_frames: Sequence[np.ndarray]) -> np.ndarray: ...  # (B, D)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...  # (T, D)


def clip_cache_version(params: ClipParams, clip_fps: float, weights_sha256: str) -> str:
    """Cache version: algorithm version + hash of parameters, sampling rate and weights.

    `batch_size` is excluded: it changes speed, not the embeddings.
    """
    p = {k: v for k, v in asdict(params).items() if k != "batch_size"}
    blob = json.dumps({**p, "clip_fps": clip_fps, "weights_sha256": weights_sha256}, sort_keys=True)
    return f"{_ALGO_VERSION}-{hashlib.sha1(blob.encode()).hexdigest()[:12]}"


# --- pure math ----------------------------------------------------------------------


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unit-length rows; all-zero rows stay zero."""
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)


def class_prompts(classes: Mapping[str, str], templates: Sequence[str]) -> list[list[str]]:
    """One list of prompts per class, in `classes` order: template.format(phrase)."""
    return [[t.format(phrase) for t in templates] for phrase in classes.values()]


def ensemble(prompt_embs: np.ndarray) -> np.ndarray:
    """(C, P, D) per-prompt embeddings -> (C, D): normalize, mean over prompts, renormalize."""
    if prompt_embs.ndim != 3:
        raise ValueError(f"expected (classes, prompts, dim), got {prompt_embs.shape}")
    return l2_normalize(l2_normalize(prompt_embs).mean(axis=1))


def zero_shot_probs(
    image_embs: np.ndarray, class_embs: np.ndarray, logit_scale: float
) -> np.ndarray:
    """(M, D) images x (C, D) classes -> (M, C) softmax over `logit_scale * cosine`."""
    logits = logit_scale * (l2_normalize(image_embs) @ l2_normalize(class_embs).T)
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def class_embeddings(
    encoder: Encoder, classes: Mapping[str, str], templates: Sequence[str]
) -> np.ndarray:
    """(C, D) prompt-ensembled text embeddings, rows in `classes` order."""
    prompts = class_prompts(classes, templates)
    flat = encoder.encode_texts([p for group in prompts for p in group])
    return ensemble(flat.reshape(len(prompts), len(templates), -1))


# --- signals (I/O edge) -------------------------------------------------------------


class ClipOnline:
    """Incremental CLIP embeddings: `add` one (t, BGR frame) at a time, then `result`.

    Frames are encoded in batches of `batch_size` as they arrive.
    """

    def __init__(self, encoder: Encoder, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._encoder = encoder
        self._batch_size = batch_size
        self._times: list[float] = []
        self._chunks: list[np.ndarray] = []
        self._batch: list[np.ndarray] = []

    def add(self, t: float, bgr: np.ndarray) -> None:
        self._times.append(t)
        self._batch.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if len(self._batch) == self._batch_size:
            self._chunks.append(self._encoder.encode_images(self._batch))
            self._batch = []

    def result(self) -> dict[str, np.ndarray]:
        if self._batch:
            self._chunks.append(self._encoder.encode_images(self._batch))
            self._batch = []
        if not self._times:
            raise ValueError("no frames decoded")
        emb = l2_normalize(np.concatenate(self._chunks)).astype(np.float32)
        return {"t": np.asarray(self._times, dtype=np.float64), "clip_emb": emb}


def clip_signals(
    frames: Iterable[tuple[float, np.ndarray]], encoder: Encoder, batch_size: int
) -> dict[str, np.ndarray]:
    """{"t": (M,), "clip_emb": (M, D) float32, L2-normalized} for (t, BGR frame) pairs."""
    online = ClipOnline(encoder, batch_size)
    for t, bgr in frames:
        online.add(t, bgr)
    return online.result()


class OpenClipEncoder:
    """open_clip model on CPU, loaded from a local weights file (no network)."""

    def __init__(self, params: ClipParams, repo_root: Path) -> None:
        # Imported here so the default test suite doesn't load torch.
        import open_clip
        import torch
        from PIL import Image

        weights = repo_root / params.weights
        if not weights.exists():
            raise FileNotFoundError(f"{weights} missing: run tools/fetch_weights.sh")
        torch.manual_seed(0)
        model, _, preprocess = open_clip.create_model_and_transforms(
            params.model, pretrained=str(weights), quick_gelu=True, device="cpu"
        )
        model.eval()
        self._torch = torch
        self._image = Image
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(params.model)
        self._width = params.width
        self.logit_scale = float(model.logit_scale.detach().exp())

    def encode_images(self, rgb_frames: Sequence[np.ndarray]) -> np.ndarray:
        imgs = [
            self._preprocess(self._image.fromarray(resize_to_width(f, self._width)))
            for f in rgb_frames
        ]
        with self._torch.no_grad():
            out = self._model.encode_image(self._torch.stack(imgs))
        return l2_normalize(out.numpy()).astype(np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        with self._torch.no_grad():
            out = self._model.encode_text(self._tokenizer(list(texts)))
        return l2_normalize(out.numpy()).astype(np.float32)
