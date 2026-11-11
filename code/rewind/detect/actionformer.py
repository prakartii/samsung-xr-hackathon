"""ActionFormer edge: SlowFast features -> verb or noun detections (ARCHITECTURE.md §3.1).

Runs the vendored ActionFormer (code/third_party/actionformer) on CPU with the
released EPIC-100 checkpoints, one model for verbs and one for nouns. Each model
returns (segment in seconds, class index, score) after its own soft-NMS and
grid-to-seconds conversion (meta_archs.postprocessing). Class index i is kit key i
(kit vocabulary = EPIC class ids in order + `other`; checked by build_vn_prior.py).

`to_detections` (index -> key, clamping, ordering) is pure; `ActionFormerDetector`
loads torch and the vendored code only when constructed.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rewind.detect.pairing import Detection

# Released EPIC features (ActionFormer README / config): 32-frame SlowFast clips at
# 30 fps with a stride of 16 frames -> one feature per 16/30 s. These describe the
# released features and checkpoints, so they are facts of the model, not knobs.
FEAT_FPS = 30.0
FEAT_STRIDE = 16
FEAT_NUM_FRAMES = 32
FEAT_DIM = 2304


def feature_times(n: int) -> np.ndarray:
    """Center time (s) of each feature: (i * stride + num_frames / 2) / fps."""
    return (np.arange(n) * FEAT_STRIDE + 0.5 * FEAT_NUM_FRAMES) / FEAT_FPS


def to_detections(
    segments: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    keys: Sequence[str],
    duration: float,
) -> list[Detection]:
    """Model output -> [(segment, key, score)], highest score first.

    Segments are clamped to [0, duration]; empty ones (start >= end after clamping)
    and labels outside `keys` are dropped. Ties sort by start, then key.
    """
    out = []
    for (s, e), p, lab in zip(segments.tolist(), scores.tolist(), labels.tolist(), strict=True):
        s, e = max(0.0, float(s)), min(float(duration), float(e))
        if e <= s or not 0 <= int(lab) < len(keys):
            continue
        out.append(((s, e), keys[int(lab)], float(np.clip(p, 0.0, 1.0))))
    out.sort(key=lambda d: (-d[2], d[0][0], d[1]))
    return out


@dataclass(frozen=True)
class ActionFormerParams:
    config: str  # vendored yaml, e.g. code/third_party/actionformer/configs/epic_slowfast_verb.yaml
    checkpoint: str  # e.g. weights/actionformer_epic_verb.pth.tar


class ActionFormerDetector:
    """One ActionFormer model (verbs or nouns) on CPU, EMA weights from the checkpoint."""

    def __init__(self, params: ActionFormerParams, keys: Sequence[str], repo_root: Path) -> None:
        import torch

        vendored = str(repo_root / "code" / "third_party" / "actionformer")
        if vendored not in sys.path:
            sys.path.insert(0, vendored)
        from libs.core import load_config
        from libs.modeling import make_meta_arch

        ckpt_path = repo_root / params.checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"{ckpt_path} missing: run tools/fetch_weights.sh")
        cfg = load_config(str(repo_root / params.config))
        model = make_meta_arch(cfg["model_name"], **cfg["model"])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = {k.removeprefix("module."): v for k, v in ckpt["state_dict_ema"].items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        self._torch = torch
        self._model = model
        self._keys = tuple(keys)

    def detect(self, feats: np.ndarray, duration: float) -> list[Detection]:
        """feats: (T, 2304) released-format SlowFast features for one video."""
        if feats.ndim != 2 or feats.shape[1] != FEAT_DIM:
            raise ValueError(f"expected (T, {FEAT_DIM}) features, got {feats.shape}")
        item = {
            "video_id": "v",  # internal to the model call; never derived from the filename
            "feats": self._torch.from_numpy(np.ascontiguousarray(feats.T, dtype=np.float32)),
            "segments": None,
            "labels": None,
            "fps": FEAT_FPS,
            "duration": float(duration),
            "feat_stride": FEAT_STRIDE,
            "feat_num_frames": FEAT_NUM_FRAMES,
        }
        with self._torch.no_grad():
            (res,) = self._model([item])
        return to_detections(
            res["segments"].numpy(),
            res["scores"].numpy(),
            res["labels"].numpy(),
            self._keys,
            duration,
        )
