"""Video I/O: the only place frames are converted to seconds and back.

CLAUDE.md §6.4: time is float seconds everywhere; frame <-> seconds conversion lives
only in this module. The conversion helpers are pure; ffprobe and decoding are the
I/O edge.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Absorbs float error when a timestamp sits exactly on a frame boundary, e.g.
# 1001 / (30000/1001) * (30000/1001) evaluating to 1000.9999999. Far below one frame
# at any real frame rate (1/240 s), so it never moves a timestamp to the next frame.
_FRAME_EPS = 1e-6


def parse_rate(rate: str) -> float:
    """Parse an ffprobe rate string ("30000/1001", "25/1", "25") into frames per second."""
    value = float(Fraction(rate))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid frame rate: {rate!r}")
    return value


def frame_to_sec(frame_idx: int, fps: float) -> float:
    """Start time in seconds of frame `frame_idx` (0-based)."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if frame_idx < 0:
        raise ValueError(f"frame index must be >= 0, got {frame_idx}")
    return frame_idx / fps


def sec_to_frame(t: float, fps: float) -> int:
    """Index of the frame shown at time `t` seconds (the frame whose interval contains t)."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if t < 0:
        raise ValueError(f"time must be >= 0, got {t}")
    return math.floor(t * fps + _FRAME_EPS)


def sample_times(duration_sec: float, target_fps: float) -> np.ndarray:
    """Timestamps k / target_fps for every k with k / target_fps < duration_sec.

    This is the time grid every per-frame signal is sampled on, so all signals at the
    same rate share identical timestamps.
    """
    if duration_sec <= 0:
        raise ValueError(f"duration must be positive, got {duration_sec}")
    if target_fps <= 0:
        raise ValueError(f"target fps must be positive, got {target_fps}")
    n = math.ceil(duration_sec * target_fps - _FRAME_EPS)
    return np.arange(n, dtype=np.float64) / target_fps


# --- ffprobe (I/O edge) -------------------------------------------------------------


@dataclass(frozen=True)
class ProbeMeta:
    """What ffprobe tells us about a video. `pipeline` builds `types.VideoMeta` from it."""

    duration_sec: float
    fps: float
    width: int
    height: int


def parse_ffprobe(probe: dict[str, Any]) -> ProbeMeta:
    """Extract duration, fps and size from `ffprobe -show_format -show_streams` JSON.

    Duration: the container's `format.duration`, else the video stream's. FPS: the
    stream's `avg_frame_rate` (true average, robust to variable frame rate), else
    `r_frame_rate` when avg is missing or "0/0".
    """
    streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise ValueError("no video stream in ffprobe output")
    video = streams[0]

    raw_duration = probe.get("format", {}).get("duration") or video.get("duration")
    if raw_duration is None:
        raise ValueError("ffprobe output has no duration")
    duration = float(raw_duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid duration: {raw_duration!r}")

    fps = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = video.get(key)
        if rate and rate != "0/0":
            fps = parse_rate(rate)
            break
    if fps is None:
        raise ValueError("ffprobe output has no frame rate")

    return ProbeMeta(
        duration_sec=duration,
        fps=fps,
        width=int(video["width"]),
        height=int(video["height"]),
    )


def ffprobe_meta(path: Path) -> ProbeMeta:
    """Run ffprobe on `path` and parse its metadata. Needs `ffprobe` on PATH."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return parse_ffprobe(json.loads(out))


# --- decoding (I/O edge) ------------------------------------------------------------


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    """Resize keeping the aspect ratio; height is rounded to an even number (>= 2)."""
    h, w = frame.shape[:2]
    if w == width:
        return frame
    height = max(2, 2 * round(h * width / w / 2))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def iter_frames(
    path: Path,
    duration_sec: float,
    fps: float,
    target_fps: float,
    width: int | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t, BGR frame) at every t in `sample_times(duration_sec, target_fps)`.

    Each t gets the source frame shown at t (`sec_to_frame(t, fps)`), so t is exact
    and shared by every signal sampled at the same rate. `duration_sec` and `fps` come
    from ffprobe (`ProbeMeta`) so all modules agree on them. Frames are decoded
    sequentially (grab skips without decoding pixels), which beats seeking for
    sampling rates of a few fps. If the decoder runs out early, iteration stops; the
    caller compares the count with its timestamps.
    """
    times = sample_times(duration_sec, target_fps)
    targets = [sec_to_frame(float(t), fps) for t in times]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    try:
        current = -1  # index of the last frame grabbed
        frame = None
        for t, target in zip(times, targets, strict=True):
            while current < target:
                if not cap.grab():
                    return
                current += 1
                frame = None
            if frame is None:
                ok, frame = cap.retrieve()
                if not ok:
                    return
                if width is not None:
                    frame = resize_to_width(frame, width)
            yield float(t), frame
    finally:
        cap.release()
