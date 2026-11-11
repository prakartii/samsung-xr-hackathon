"""Tiny synthetic videos built in code (CLAUDE.md §7.3.9: no committed binaries).

Uses OpenCV's writer so the default suite needs no ffmpeg on PATH.
"""

from pathlib import Path

import cv2
import numpy as np


def write_index_video(path: Path, n_frames: int, fps: float, size=(64, 48)) -> Path:
    """Write an MJPG .avi where frame k is a flat gray of intensity 8 * k (k < 32).

    The intensity identifies the frame after lossy decoding (steps of 8 >> JPEG error).
    """
    if n_frames > 32:
        raise ValueError("intensity 8 * k must stay <= 255")
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not writer.isOpened():
        raise OSError("OpenCV cannot write MJPG video")
    for k in range(n_frames):
        writer.write(np.full((h, w, 3), 8 * k, dtype=np.uint8))
    writer.release()
    return path


def frame_index(frame: np.ndarray) -> int:
    """Recover k from a frame written by `write_index_video`."""
    return round(float(frame.mean()) / 8)
