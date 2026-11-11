"""Decoding at a target fps on a tiny generated video (no ffmpeg, no data/)."""

import shutil
import subprocess

import numpy as np
import pytest
from fixtures.videos import frame_index, write_index_video

from rewind.video_io import DecodeError, iter_frames, probe_video, resize_to_width


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
