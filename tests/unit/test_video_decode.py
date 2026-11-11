"""Decoding at a target fps on a tiny generated video (no ffmpeg, no data/)."""

import numpy as np
import pytest

from fixtures.videos import frame_index, write_index_video
from rewind.video_io import iter_frames, resize_to_width


@pytest.fixture
def video_10fps_3s(tmp_path):
    # 30 frames at 10 fps = 3.0 s; frame k has intensity 8k.
    return write_index_video(tmp_path / "v.avi", n_frames=30, fps=10.0)


def test_iter_frames_2fps_picks_every_fifth_frame(video_10fps_3s):
    # t = 0.0, 0.5, ..., 2.5 -> frame = floor(t * 10) = 0, 5, 10, 15, 20, 25
    out = list(iter_frames(video_10fps_3s, duration_sec=3.0, fps=10.0, target_fps=2.0))
    assert [t for t, _ in out] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], abs=1e-12)
    assert [frame_index(f) for _, f in out] == [0, 5, 10, 15, 20, 25]


def test_iter_frames_above_source_fps_repeats_frames(video_10fps_3s):
    # target 20 fps over the first 0.3 s: t = 0, .05, .1, .15, .2, .25
    # -> frames floor(10 t) = 0, 0, 1, 1, 2, 2
    out = list(iter_frames(video_10fps_3s, duration_sec=0.3, fps=10.0, target_fps=20.0))
    assert [frame_index(f) for _, f in out] == [0, 0, 1, 1, 2, 2]


def test_iter_frames_stops_when_video_ends_early(video_10fps_3s):
    # Claimed duration 5 s but only 30 frames (3 s): t = 0..4.5 at 2 fps would need
    # frames up to 45; decoding stops after frame 25 (t = 2.5) -> 6 frames.
    out = list(iter_frames(video_10fps_3s, duration_sec=5.0, fps=10.0, target_fps=2.0))
    assert len(out) == 6


def test_iter_frames_resizes_to_width(video_10fps_3s):
    # 64x48 -> width 32: height = 48 * 32 / 64 = 24
    _, frame = next(iter_frames(video_10fps_3s, 3.0, 10.0, 2.0, width=32))
    assert frame.shape[:2] == (24, 32)


def test_iter_frames_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        next(iter_frames(tmp_path / "nope.mp4", 3.0, 10.0, 2.0))


def test_resize_to_width_keeps_even_height():
    # 100x75 -> width 160: 75 * 160 / 100 = 120 (already even)
    assert resize_to_width(np.zeros((75, 100, 3), np.uint8), 160).shape[:2] == (120, 160)
    # 30x7 -> width 10: 7 * 10 / 30 = 2.33 -> nearest even = 2
    assert resize_to_width(np.zeros((7, 30, 3), np.uint8), 10).shape[:2] == (2, 10)
