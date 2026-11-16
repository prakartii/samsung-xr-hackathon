"""Video metadata and decoding. The only place frames <-> seconds are converted.

CLAUDE.md §6.4: time is float seconds everywhere. P1 (B) added ffprobe metadata;
P3 (A) adds frame/seconds conversion, the shared sampling grid and decoding with PyAV
(PROGRESS.md D1: OpenCV stopped silently on EPIC GoPro files).
"""

import json
import math
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


class VideoError(RuntimeError):
    """The video can't be read (missing, corrupt, no video stream)."""


@dataclass(frozen=True)
class ProbeInfo:
    duration_sec: float
    fps: float
    width: int
    height: int


def ffprobe_meta(path: str | Path) -> ProbeInfo:
    """Read duration, fps and size with ffprobe. Raises VideoError on failure.

    No ffprobe binary on PATH → `probe_video` (PyAV, same duration rules), so a
    machine without the ffmpeg CLI still gets real output instead of none.
    """
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
    except FileNotFoundError:
        return probe_video(path)
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
    if not (math.isfinite(duration) and duration > 0):
        raise VideoError(f"{where}: duration {duration} is not a positive finite number")
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


# --- frame <-> seconds (pure) -----------------------------------------------------

# Absorbs float error when a timestamp sits exactly on a frame boundary, e.g.
# 1001 / (30000/1001) * (30000/1001) evaluating to 1000.9999999. Far below one frame
# at any real frame rate (1/240 s), so it never moves a timestamp to the next frame.
_FRAME_EPS = 1e-6

# How far past the last video frame a requested sample may fall and still get that
# frame. Container durations (which include audio) can run a few hundredths of a
# second past the video stream (0.03 s on one EPIC video). Real truncation is seconds to
# minutes short (OpenCV stopped after 0.7 s of 174 s), so 0.25 s still catches it.
_END_SLACK_SEC = 0.25


def parse_rate(rate: str) -> float:
    """Parse a rate string ("30000/1001", "25/1", "25") into frames per second."""
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


# --- decoding (I/O edge) ------------------------------------------------------------


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    """Resize keeping the aspect ratio; height is rounded to an even number (>= 2)."""
    h, w = frame.shape[:2]
    if w == width:
        return frame
    height = max(2, 2 * round(h * width / w / 2))
    import cv2  # lazy: run.sh's metadata/fallback path must not load OpenCV

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


class DecodeError(RuntimeError):
    """The decoder ended before the requested duration (truncated or unreadable file)."""


def probe_video(path: str | Path) -> ProbeInfo:
    """ProbeInfo via PyAV (the FFmpeg libraries used for decoding; no ffprobe binary).

    Same rules as `parse_probe`: container duration first, else the stream's; raises
    VideoError when there is no video stream or no usable duration.
    """
    import av

    try:
        container = av.open(str(path))
    except (OSError, av.error.FFmpegError) as exc:
        raise VideoError(f"cannot open {path}: {exc}") from exc
    with container:
        if not container.streams.video:
            raise VideoError(f"{path}: no video stream")
        stream = container.streams.video[0]
        if container.duration is not None:
            duration = container.duration / 1_000_000  # microseconds
        elif stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        else:
            raise VideoError(f"{path}: no usable duration")
        if not (math.isfinite(duration) and duration > 0):
            raise VideoError(f"{path}: duration {duration} is not a positive finite number")
        fps = float(stream.average_rate or stream.base_rate or 0)
        return ProbeInfo(duration, fps, stream.codec_context.width, stream.codec_context.height)


# Undecodable packets tolerated per video before giving up: about 1 s at 60 fps. Two
# EPIC videos (MD5-verified, so genuine recordings) each contain a packet that
# FFmpeg rejects ("Invalid data found when processing input"); one bad chunk must not
# cost the whole video, but a badly damaged file should still fail loudly.
_MAX_BAD_PACKETS = 60


def frames_skipping_bad_packets(packets, max_bad: int = _MAX_BAD_PACKETS):
    """Decoded frames from demuxed packets, skipping packets the decoder rejects.

    Raises DecodeError after more than `max_bad` rejected packets.
    """
    import av

    bad = 0
    for packet in packets:
        try:
            frames = packet.decode()
        except av.error.InvalidDataError as exc:
            bad += 1
            if bad > max_bad:
                raise DecodeError(f"more than {max_bad} undecodable packets ({exc})") from exc
            continue
        yield from frames


def iter_frames(
    path: Path,
    duration_sec: float,
    target_fps: float,
    width: int | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t, BGR frame) at every t in `sample_times(duration_sec, target_fps)`.

    Each t gets the frame shown at t: the last decoded frame whose presentation time is
    <= t (the first frame for t before it). Timestamps come from the container, so this
    is exact for variable frame rate too, and t is shared by every signal sampled at
    the same rate.

    Decoding uses PyAV with threaded decode of the first video stream only; other
    streams (audio, GoPro metadata) are ignored. OpenCV's backend stopped silently
    after < 1 s on EPIC GoPro files, which is why this module doesn't use it.

    Packets the decoder rejects are skipped (up to `_MAX_BAD_PACKETS`); samples in
    the gap get the last frame decoded before it.

    Raises DecodeError if the stream ends more than one frame plus `_END_SLACK_SEC`
    before the last requested t, so a truncated decode can never be mistaken for a
    short video.
    """
    import av

    times = sample_times(duration_sec, target_fps)
    n = len(times)
    idx = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frame_dur = 1.0 / float(stream.average_rate or stream.base_rate or 30)
        # Presentation times are absolute; the sample grid starts at 0. Without this a
        # stream that starts at 2 s (e.g. -output_ts_offset, some GoPro remuxes) would
        # be sampled 2 s late and silently misaligned with every action time.
        t0 = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
        prev = None  # (time, av.VideoFrame)
        cached: tuple[object, np.ndarray] | None = None  # (frame, its image); holds a ref

        def image(frame) -> np.ndarray:
            nonlocal cached
            if cached is not None and cached[0] is frame:
                return cached[1]
            if width is None:
                img = frame.to_ndarray(format="bgr24")
            else:
                h = max(2, 2 * round(frame.height * width / frame.width / 2))
                img = frame.reformat(width=width, height=h, format="bgr24").to_ndarray()
            cached = (frame, img)
            return img

        for frame in frames_skipping_bad_packets(container.demux(stream)):
            ft = (
                frame.time - t0
                if frame.time is not None
                else (prev[0] + frame_dur if prev else 0.0)
            )
            while idx < n and ft > times[idx] + _FRAME_EPS:
                yield float(times[idx]), image(prev[1] if prev is not None else frame)
                idx += 1
            if idx >= n:
                return
            prev = (ft, frame)
        if prev is None:
            raise DecodeError(f"no video frames decoded from {path}")
        # remaining targets inside the last frame's display interval get the last frame
        while idx < n and times[idx] <= prev[0] + frame_dur + _END_SLACK_SEC:
            yield float(times[idx]), image(prev[1])
            idx += 1
        if idx < n:
            raise DecodeError(
                f"{path}: stream ends at {prev[0]:.2f} s but samples were requested up to "
                f"{times[-1]:.2f} s ({n - idx} short)"
            )
