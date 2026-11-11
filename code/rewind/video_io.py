"""Video metadata and decoding. The only place frames ↔ seconds are converted.

P1 has metadata only; P3 (A) adds frame decoding.
"""

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class VideoError(RuntimeError):
    """The video can't be read (missing, corrupt, no video stream)."""


@dataclass(frozen=True)
class ProbeInfo:
    duration_sec: float
    fps: float
    width: int
    height: int


def ffprobe_meta(path: str | Path) -> ProbeInfo:
    """Read duration, fps and size with ffprobe. Raises VideoError on failure."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        info = json.loads(out.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise VideoError(f"ffprobe failed on {path}: {exc}") from exc
    return parse_probe(info, str(path))


def parse_probe(info: dict, where: str = "<probe>") -> ProbeInfo:
    """ProbeInfo from ffprobe's JSON. Pure. Prefers the container duration."""
    streams = info.get("streams") or []
    if not streams:
        raise VideoError(f"{where}: no video stream")
    st = streams[0]
    raw = (info.get("format") or {}).get("duration") or st.get("duration")
    try:
        duration = float(raw)
    except (TypeError, ValueError) as exc:
        raise VideoError(f"{where}: no usable duration ({raw!r})") from exc
    if not duration > 0:
        raise VideoError(f"{where}: duration {duration} is not positive")
    try:
        fps = float(Fraction(st.get("avg_frame_rate") or "0/1"))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return ProbeInfo(
        duration_sec=duration,
        fps=fps,
        width=int(st.get("width") or 0),
        height=int(st.get("height") or 0),
    )
