"""Real MediaPipe Hand Landmarker (tier: data). Run with `pytest -m data`.

Needs weights/hand_landmarker.task (tools/fetch_weights.sh); the EPIC check also
needs data/videos/P01_11.mp4 (tools/download_videos.py). Skips when either is missing.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from rewind.features.hands import HandParams, MediaPipeHands, hands_signals
from rewind.video_io import iter_frames

pytestmark = pytest.mark.data

REPO = Path(__file__).resolve().parents[2]
PARAMS = HandParams(
    model="weights/hand_landmarker.task",
    width=320,
    num_hands=2,
    min_detection_conf=0.3,
    min_presence_conf=0.3,
    min_tracking_conf=0.3,
)
EPIC_VIDEO = REPO / "data" / "videos" / "P01_11.mp4"


@pytest.fixture
def scorer():
    if not (REPO / PARAMS.model).exists():
        pytest.skip("weights/hand_landmarker.task missing: run tools/fetch_weights.sh")
    with MediaPipeHands(PARAMS, REPO) as s:
        yield s


def test_mediapipe_blank_frames_have_no_hands(scorer):
    frames = [(k / 4, np.zeros((240, 320, 3), np.uint8)) for k in range(8)]
    out = hands_signals(frames, scorer)
    np.testing.assert_array_equal(out["hands"], np.zeros(8))


def test_mediapipe_epic_kitchen_clip_mostly_shows_hands(scorer):
    # EPIC is egocentric cooking: the wearer's hands are in view most of the time.
    # Over the first 30 s at 2 fps (60 frames), expect hands in at least half.
    if not EPIC_VIDEO.exists():
        pytest.skip(f"{EPIC_VIDEO} missing: run tools/download_videos.py")
    cap = cv2.VideoCapture(str(EPIC_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)  # EPIC is mostly 59.94, but not all videos
    cap.release()
    frames = iter_frames(EPIC_VIDEO, duration_sec=30.0, fps=fps, target_fps=2.0, width=640)
    out = hands_signals(frames, scorer)
    assert len(out["hands"]) == 60
    assert np.mean(out["hands"] > 0) >= 0.5
