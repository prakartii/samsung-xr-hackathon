"""Real open_clip ViT-B/16 (tier: data). Run with `pytest -m data`.

Needs weights/clip_vit_b16_openai.safetensors (tools/fetch_weights.sh) and, for the
routing check, FPVSum videos in data/videos (tools/download_videos.py).
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from rewind.config import load_config
from rewind.features.clip import (
    ClipParams,
    OpenClipEncoder,
    class_embeddings,
    clip_signals,
    zero_shot_probs,
)
from rewind.video_io import iter_frames, probe_video

pytestmark = pytest.mark.data

REPO = Path(__file__).resolve().parents[2]
CFG = load_config(REPO / "configs" / "default.yaml")
PARAMS = ClipParams(**asdict(CFG.features.clip))
CLASSES = CFG.label.classes
TEMPLATES = CFG.label.prompts


@pytest.fixture(scope="module")
def encoder():
    if not (REPO / PARAMS.weights).exists():
        pytest.skip(f"{PARAMS.weights} missing: run tools/fetch_weights.sh")
    return OpenClipEncoder(PARAMS, REPO)


def _keyframe_embeddings(encoder, video: Path, n: int = 8) -> np.ndarray:
    duration = probe_video(video).duration_sec
    return clip_signals(iter_frames(video, duration, n / duration), encoder, n)["clip_emb"]


def test_openclip_embeddings_are_unit_float32(encoder):
    emb = encoder.encode_images([np.zeros((240, 320, 3), np.uint8)] * 2)
    assert emb.shape == (2, 512)
    assert emb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_openclip_is_deterministic(encoder):
    img = [np.full((240, 320, 3), 128, np.uint8)]
    np.testing.assert_array_equal(encoder.encode_images(img), encoder.encode_images(img))


@pytest.mark.parametrize("prefix", ["Ski_", "Scuba_", "Surfing_"])
def test_openclip_fpvsum_sports_video_is_not_cooking(encoder, prefix):
    # Routing (ARCHITECTURE.md §6.1): mean P(cooking) over 8 keyframes must be far
    # below the 0.5 threshold. Measured 2026-11-11: <= 0.006 on 9 FPVSum videos.
    # (The filename only picks a test video; inference never sees it.)
    videos = sorted((REPO / "data" / "videos").glob(f"{prefix}*.mp4"))
    if not videos:
        pytest.skip(f"no {prefix}* video in data/videos: run tools/download_videos.py")
    cls = class_embeddings(encoder, CLASSES, TEMPLATES)
    probs = zero_shot_probs(_keyframe_embeddings(encoder, videos[0]), cls, encoder.logit_scale)
    assert probs[:, list(CLASSES).index("cooking")].mean() < 0.1
