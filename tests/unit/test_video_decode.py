"""Decoding at a target fps on a tiny generated video (no ffmpeg, no data/)."""

import shutil
import subprocess

import av
import numpy as np
import pytest
from fixtures.videos import corrupt_frame, frame_index, write_index_video

from rewind import video_io
from rewind.video_io import (
    DecodeError,
    frames_skipping_bad_packets,
    iter_frames,
    probe_video,
    resize_to_width,
)


@pytest.fixture
def video_10fps_3s(tmp_path):
    # 30 frames at 10 fps = 3.0 s; frame k has intensity 8k.
    return write_index_video(tmp_path / "v.avi", n_frames=30, fps=10.0)


def test_iter_frames_2fps_picks_every_fifth_frame(video_10fps_3s):
    # t = 0.0, 0.5, ..., 2.5 -> frame = floor(t * 10) = 0, 5, 10, 15, 20, 25
    out = list(iter_frames(video_10fps_3s, duration_sec=3.0, target_fps=2.0))
    assert [t for t, _ in out] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], abs=1e-12)
    assert [frame_index(f) for _, f in out] == [0, 5, 10, 15, 20, 25]


def test_iter_frames_above_source_fps_repeats_frames(video_10fps_3s):
    # target 20 fps over the first 0.3 s: t = 0, .05, .1, .15, .2, .25
    # -> frames floor(10 t) = 0, 0, 1, 1, 2, 2
    out = list(iter_frames(video_10fps_3s, duration_sec=0.3, target_fps=20.0))
    assert [frame_index(f) for _, f in out] == [0, 0, 1, 1, 2, 2]


def test_iter_frames_raises_when_video_ends_early(video_10fps_3s):
    # Claimed duration 5 s but the file holds 30 frames (last shown 2.9-3.0 s): samples
    # up to 4.5 s were requested, so the decoder ran out early -> DecodeError, never a
    # silently shorter signal (OpenCV truncated EPIC files this way).
    with pytest.raises(DecodeError):
        list(iter_frames(video_10fps_3s, duration_sec=5.0, target_fps=2.0))


def test_iter_frames_container_slightly_longer_than_video_is_not_an_error(video_10fps_3s):
    # Regression (Kaggle, EPIC P08_09): the container duration (from audio) ran 0.03 s
    # past the last video frame and iter_frames raised "1 short". Here: frames cover
    # [0, 3.0) s; duration 3.15 at 7 fps asks for t = 22/7 = 3.1429 s, 0.143 s past
    # the end -> within the end slack, so it gets the last frame (k = 29).
    out = list(iter_frames(video_10fps_3s, duration_sec=3.15, target_fps=7.0))
    assert len(out) == 23  # ceil(3.15 * 7) = ceil(22.05) = 23 samples
    assert frame_index(out[-1][1]) == 29


def test_iter_frames_last_sample_inside_last_frame_is_kept(video_10fps_3s):
    # Duration 3.0 s at 4 fps: t = 0 .. 2.75; frame 29 is shown during [2.9, 3.0),
    # so t = 2.75 gets frame 27 and nothing raises.
    out = list(iter_frames(video_10fps_3s, duration_sec=3.0, target_fps=4.0))
    assert frame_index(out[-1][1]) == 27


def test_probe_video_reads_duration_fps_and_size(video_10fps_3s):
    # 30 frames at 10 fps = 3.0 s, 64x48
    meta = probe_video(video_10fps_3s)
    assert meta.duration_sec == pytest.approx(3.0, abs=0.11)
    assert meta.fps == pytest.approx(10.0, abs=1e-6)
    assert (meta.width, meta.height) == (64, 48)


def test_iter_frames_resizes_to_width(video_10fps_3s):
    # 64x48 -> width 32: height = 48 * 32 / 64 = 24
    _, frame = next(iter_frames(video_10fps_3s, 3.0, 2.0, width=32))
    assert frame.shape[:2] == (24, 32)


def test_iter_frames_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        next(iter_frames(tmp_path / "nope.mp4", 3.0, 2.0))


def test_resize_to_width_keeps_even_height():
    # 100x75 -> width 160: 75 * 160 / 100 = 120 (already even)
    assert resize_to_width(np.zeros((75, 100, 3), np.uint8), 160).shape[:2] == (120, 160)
    # 30x7 -> width 10: 7 * 10 / 30 = 2.33 -> nearest even = 2
    assert resize_to_width(np.zeros((7, 30, 3), np.uint8), 10).shape[:2] == (2, 10)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_iter_frames_ignores_a_nonzero_stream_start_time(video_10fps_3s, tmp_path):
    # Same 30 frames, but the stream's timestamps start at 2.0 s. Sampling is relative
    # to the stream start, so at 2 fps we still get frames 0, 5, ..., 25 (an absolute
    # clock would give 0, 0, 0, 0, 0, 5: everything 2 s late).
    off = tmp_path / "off.mkv"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_10fps_3s),
         "-c", "copy", "-output_ts_offset", "2", "-y", str(off)],
        check=True,
    )  # fmt: skip

    out = list(iter_frames(off, duration_sec=3.0, target_fps=2.0))

    assert [frame_index(f) for _, f in out] == [0, 5, 10, 15, 20, 25]


# --- packets the decoder rejects (Kaggle: EPIC P29_05, P30_08) ------------------------


class _Packet:
    """Stands in for av.Packet: decode() returns frames or raises like FFmpeg does."""

    def __init__(self, frames=(), bad=False):
        self._frames, self._bad = list(frames), bad

    def decode(self):
        if self._bad:
            raise av.error.InvalidDataError(1094995529, "Invalid data found when processing input")
        return self._frames


def test_bad_packet_is_skipped_and_decoding_continues():
    # Regression: one undecodable packet (avcodec_send_packet: Invalid data) aborted the
    # whole video on Kaggle. Frames before and after it must still come through.
    packets = [_Packet(["f0"]), _Packet(bad=True), _Packet(["f1", "f2"])]
    assert list(frames_skipping_bad_packets(packets, max_bad=5)) == ["f0", "f1", "f2"]


def test_iter_frames_undecodable_frame_gets_the_previous_frame(video_10fps_3s):
    # End to end on a real file: frame 10's packet is zeroed, so FFmpeg rejects it.
    # 10 fps samples t = 0.0 .. 2.9 map to frames 0 .. 29; t = 1.0 falls in the hole
    # and gets frame 9 (the last decoded one); every other sample is exact.
    corrupt_frame(video_10fps_3s, 10)
    out = list(iter_frames(video_10fps_3s, duration_sec=3.0, target_fps=10.0))
    assert [frame_index(f) for _, f in out] == [*range(10), 9, *range(11, 30)]


def test_too_many_bad_packets_raise_decode_error():
    # More than max_bad rejected packets means a damaged file, not one bad chunk.
    packets = [_Packet(bad=True)] * 3 + [_Packet(["f0"])]
    with pytest.raises(DecodeError):
        list(frames_skipping_bad_packets(packets, max_bad=2))


def test_ffprobe_meta_without_ffprobe_binary_uses_pyav(video_10fps_3s, monkeypatch):
    # A machine with no ffprobe on PATH: subprocess.run raises FileNotFoundError.
    # The probe must still succeed via PyAV: 30 frames at 10 fps = 3.0 s, 64x48.
    def no_binary(*args, **kwargs):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(video_io.subprocess, "run", no_binary)
    meta = video_io.ffprobe_meta(video_10fps_3s)
    assert meta.duration_sec == pytest.approx(3.0, abs=0.11)
    assert meta.fps == pytest.approx(10.0, abs=1e-6)
    assert (meta.width, meta.height) == (64, 48)


def test_ffprobe_meta_ffprobe_error_still_raises(tmp_path, monkeypatch):
    # ffprobe present but failing (e.g. a corrupt file) stays a VideoError.
    def fails(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffprobe")

    monkeypatch.setattr(video_io.subprocess, "run", fails)
    with pytest.raises(video_io.VideoError):
        video_io.ffprobe_meta(tmp_path / "x.mp4")
