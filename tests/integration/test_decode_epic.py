"""Regression (tier: data): decoding must cover the whole EPIC video.

Bug found 2026-11-11: OpenCV's FFmpeg backend stopped after 44 of 10,418 frames on
P01_12.mp4 (GoPro file with audio + 3 data streams) without raising, so cached
signals covered 0.7 s of a 174 s video. Needs data/videos/P01_12.mp4.
"""

from pathlib import Path

import pytest

from rewind.video_io import iter_frames

pytestmark = pytest.mark.data

VIDEO = Path(__file__).resolve().parents[2] / "data" / "videos" / "P01_12.mp4"
DURATION = 173.81  # kit/manifest.csv


@pytest.fixture
def video():
    if not VIDEO.exists():
        pytest.skip(f"{VIDEO} missing: run tools/download_videos.py")
    return VIDEO


def test_decode_epic_gopro_file_reaches_the_end(video):
    # 1 fps over 173.81 s -> t = 0, 1, ..., 173 -> 174 samples, last at 173 s
    times = [t for t, _ in iter_frames(video, DURATION, target_fps=1.0, width=64)]
    assert len(times) == 174
    assert times[-1] == pytest.approx(173.0, abs=1e-9)
