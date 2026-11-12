"""ActionFormer input features (ARCHITECTURE.md §2 `Signals.slowfast`, §10.1).

The EPIC checkpoints only work on the features they were trained on: SlowFast
pretrained on EPIC-100 train, 32-frame clips at 30 fps, stride 16 (one 2304-d vector
per 16/30 s). Two sources fill the same cache entry:

1. ActionFormer's released features, imported offline by
   tools/import_released_features.py (keyed by the MP4's content fingerprint).
2. Extraction from the MP4 with the matching SlowFast model (clean machines and gap
   videos); checked against (1) at cosine >= 0.95 per timestep (PROGRESS.md P4).

Both write `{"t": (K,), "feats": (K, 2304) float32}` under one cache version, so the
pipeline doesn't care where an entry came from.
"""

from __future__ import annotations

import numpy as np

from rewind.detect.actionformer import FEAT_DIM, feature_times
from rewind.features.cache import FeatureCache

CACHE_NAME = "slowfast"
# Identifies the feature definition (model, clip length, stride), not who computed it.
CACHE_VERSION = "slowfast-epic100-train-32f30fps-s16-v1"


def slowfast_entry(feats: np.ndarray) -> dict[str, np.ndarray]:
    """Validate (K, 2304) features and attach their center timestamps."""
    feats = np.asarray(feats)
    if feats.ndim != 2 or feats.shape[1] != FEAT_DIM or feats.shape[0] == 0:
        raise ValueError(f"expected (K>0, {FEAT_DIM}) features, got {feats.shape}")
    if not np.all(np.isfinite(feats)):
        raise ValueError("features contain NaN or inf")
    return {"t": feature_times(feats.shape[0]), "feats": feats.astype(np.float32)}


def slowfast_cache(root) -> FeatureCache:
    return FeatureCache(root, CACHE_NAME, CACHE_VERSION)


def find_slowfast(fingerprint: str, roots) -> dict[str, np.ndarray] | None:
    """The first cached entry for `fingerprint` among `roots` (None roots skipped).

    Order: the local feature cache, then the released features shipped read-only
    under weights/ (`detect.released_features`), so `run.sh` on a clean machine
    finds the same ActionFormer inputs our submitted outputs used.
    """
    for root in roots:
        if root is None:
            continue
        entry = slowfast_cache(root).get(fingerprint)
        if entry is not None:
            return entry
    return None
