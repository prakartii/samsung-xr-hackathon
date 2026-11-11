"""Hand-presence signal (ARCHITECTURE.md §2 `Signals.hands`).

Per sampled frame, the probability that the wearer's hands are visible, in [0, 1]:
the highest confidence among detected hands, 0 when none is detected. It feeds the
hedge-window activity score (§5.5) and the FPVSum importance proxy (§6.2).

The scoring is pure and works with any `HandScorer` (a callable from an RGB frame
and its timestamp to per-hand confidences), so tests use a fake. `MediaPipeHands` is
the model edge (MediaPipe Hand Landmarker in VIDEO mode, which tracks hands between
frames and needs monotonically increasing timestamps: our shared time grid).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from rewind.video_io import resize_to_width

# Bump when the algorithm changes in a way the parameters don't capture (CLAUDE.md §6.8).
# v2 (2026-11-11): PyAV decoding by presentation time; v1 used OpenCV, which truncated
# EPIC GoPro files (tests/integration/test_decode_epic.py).
_ALGO_VERSION = "hands-v2"

# (RGB frame, timestamp in seconds) -> confidence of each detected hand, in [0, 1]
HandScorer = Callable[[np.ndarray, float], Sequence[float]]


@dataclass(frozen=True)
class HandParams:
    """`features.hands` in configs/default.yaml."""

    model: str
    width: int
    num_hands: int
    min_detection_conf: float
    min_presence_conf: float
    min_tracking_conf: float


def hands_cache_version(params: HandParams, dense_fps: float, model_sha256: str) -> str:
    """Cache version: algorithm version + hash of parameters, sampling rate and weights."""
    blob = json.dumps(
        {**asdict(params), "dense_fps": dense_fps, "model_sha256": model_sha256}, sort_keys=True
    )
    return f"{_ALGO_VERSION}-{hashlib.sha1(blob.encode()).hexdigest()[:12]}"


def presence(scores: Sequence[float]) -> float:
    """Presence for one frame: max hand confidence, 0 if no hand; clipped to [0, 1]."""
    if len(scores) == 0:
        return 0.0
    return float(np.clip(max(scores), 0.0, 1.0))


class HandsOnline:
    """Incremental hand presence: `add` one (t, BGR frame) at a time, then `result`."""

    def __init__(self, scorer: HandScorer) -> None:
        self._scorer = scorer
        self._times: list[float] = []
        self._values: list[float] = []

    def add(self, t: float, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._times.append(t)
        self._values.append(presence(self._scorer(rgb, t)))

    def result(self) -> dict[str, np.ndarray]:
        if not self._times:
            raise ValueError("no frames decoded")
        return {"t": np.asarray(self._times, dtype=np.float64), "hands": np.asarray(self._values)}


def hands_signals(
    frames: Iterable[tuple[float, np.ndarray]], scorer: HandScorer
) -> dict[str, np.ndarray]:
    """{"t": (N,), "hands": (N,)} for (t, BGR frame) pairs from `iter_frames`."""
    online = HandsOnline(scorer)
    for t, bgr in frames:
        online.add(t, bgr)
    return online.result()


class MediaPipeHands:
    """MediaPipe Hand Landmarker as a `HandScorer`. Use as a context manager.

    Confidence per hand = the score of its handedness classification; a hand is
    only returned after passing the detection and presence thresholds.
    """

    def __init__(self, params: HandParams, repo_root: Path) -> None:
        # Imported here so the default test suite doesn't load MediaPipe.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model = repo_root / params.model
        if not model.exists():
            raise FileNotFoundError(f"{model} missing: run tools/fetch_weights.sh")
        self._mp = mp
        self._width = params.width
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=params.num_hands,
            min_hand_detection_confidence=params.min_detection_conf,
            min_hand_presence_confidence=params.min_presence_conf,
            min_tracking_confidence=params.min_tracking_conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def __call__(self, rgb: np.ndarray, t: float) -> list[float]:
        rgb = np.ascontiguousarray(resize_to_width(rgb, self._width))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, round(t * 1000))
        return [h[0].score for h in result.handedness if h]

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> MediaPipeHands:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
