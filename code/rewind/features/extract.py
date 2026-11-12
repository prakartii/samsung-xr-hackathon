"""One decoding pass → flow, hands and CLIP signals, through the versioned cache.

Inference edge (ARCHITECTURE.md §4 step 3): read each feature from `cache/` when a
valid entry exists (keyed by the MP4's content fingerprint, never its name), compute
the missing ones in a single decode, and store them. On a clean machine nothing is
cached and everything comes from the MP4. tools/cache_features.py batches the same
function over a directory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from rewind.config import Config
from rewind.features.cache import FeatureCache, content_fingerprint
from rewind.features.clip import ClipOnline, ClipParams, Encoder
from rewind.features.flow import FlowOnline, FlowParams, flow_cache_version
from rewind.features.hands import HandParams, HandScorer, HandsOnline
from rewind.video_io import iter_frames

FEATURES = ("flow", "hands", "clip")
Arrays = dict[str, np.ndarray]


class ExtractParams(NamedTuple):
    dense_fps: float
    clip_fps: float
    flow: FlowParams
    hands: HandParams
    clip: ClipParams

    @classmethod
    def from_config(cls, cfg: Config) -> ExtractParams:
        f = cfg.features
        return cls(
            cfg.signals.dense_fps,
            cfg.signals.clip_fps,
            FlowParams(**asdict(f.flow)),
            HandParams(**asdict(f.hands)),
            ClipParams(**asdict(f.clip)),
        )


@dataclass(frozen=True)
class SignalModels:
    """What extraction needs from the model edge.

    new_hands: a fresh hand scorer per video (MediaPipe's VIDEO mode needs increasing
    timestamps); closed after the pass if it has a ``close`` method.
    *_version: cache versions (they include each model file's hash).
    """

    clip: Encoder
    new_hands: Callable[[], HandScorer]
    clip_version: str
    hands_version: str


def compute_single_pass(
    path: Path,
    duration: float,
    missing: list[str],
    encoder: Encoder | None,
    new_hands: Callable[[], HandScorer] | None,
    p: ExtractParams,
) -> dict[str, Arrays]:
    """Decode once and feed every feature in `missing` (ARCHITECTURE.md §12 budget).

    Flow and hands use the dense grid; CLIP uses every r-th dense sample, where
    r = dense_fps / clip_fps, with its own timestamps j / clip_fps (identical to
    sample_times(T, clip_fps)). Frames are decoded at the widest size any feature
    needs; each feature downsizes from there.
    """
    need_dense = bool({"flow", "hands"} & set(missing))
    ratio = p.dense_fps / p.clip_fps
    if need_dense and abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("signals.dense_fps must be a multiple of signals.clip_fps")
    rate, step = (p.dense_fps, round(ratio)) if need_dense else (p.clip_fps, 1)
    width = max(
        [p.flow.width] * ("flow" in missing)
        + [p.hands.width] * ("hands" in missing)
        + [p.clip.width] * ("clip" in missing)
    )
    online: dict = {}
    scorer = None
    if "flow" in missing:
        online["flow"] = FlowOnline(p.flow)
    if "hands" in missing:
        if new_hands is None:
            raise ValueError("hands requested without a hand scorer")
        scorer = new_hands()
        online["hands"] = HandsOnline(scorer)
    if "clip" in missing:
        if encoder is None:
            raise ValueError("clip requested without an encoder")
        online["clip"] = ClipOnline(encoder, p.clip.batch_size)
    try:
        for k, (t, frame) in enumerate(iter_frames(path, duration, rate, width)):
            if "flow" in online:
                online["flow"].add(t, frame)
            if "hands" in online:
                online["hands"].add(t, frame)
            if "clip" in online and k % step == 0:
                online["clip"].add((k // step) / p.clip_fps, frame)
    finally:
        if scorer is not None and hasattr(scorer, "close"):
            scorer.close()
    return {feat: o.result() for feat, o in online.items()}


def load_signals(
    path: Path,
    duration: float,
    cfg: Config,
    models: SignalModels,
    cache_root: Path | None,
) -> dict[str, Arrays]:
    """{"flow": ..., "hands": ..., "clip": ...} for one video; cached when possible.

    cache_root None: compute everything and store nothing.
    """
    p = ExtractParams.from_config(cfg)
    got: dict[str, Arrays | None] = dict.fromkeys(FEATURES)
    caches: dict[str, FeatureCache] = {}
    fp = ""
    if cache_root is not None:
        fp = content_fingerprint(path)
        caches = {
            "flow": FeatureCache(cache_root, "flow", flow_cache_version(p.flow, p.dense_fps)),
            "hands": FeatureCache(cache_root, "hands", models.hands_version),
            "clip": FeatureCache(cache_root, "clip", models.clip_version),
        }
        got = {f: caches[f].get(fp) for f in FEATURES}
    missing = [f for f in FEATURES if got[f] is None]
    if missing:
        new = compute_single_pass(path, duration, missing, models.clip, models.new_hands, p)
        for feat, arrays in new.items():
            if caches:
                caches[feat].put(fp, arrays)
            got[feat] = arrays
    return {f: got[f] for f in FEATURES}  # type: ignore[misc]
