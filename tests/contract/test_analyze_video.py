"""rewind.analyze.analyze_video on real decoded videos with fake models.

Decoding, flow and the feature cache are real; CLIP and the hand model are small
deterministic fakes (CLAUDE.md §7.3 rule 6), so this runs in the default suite.
"""

import json
import shutil
import zlib
from pathlib import Path

import numpy as np
import pytest

from rewind.analyze import Models, analyze_video
from rewind.config import load_config
from rewind.features.extract import ExtractParams, SignalModels, compute_single_pass
from rewind.output import dumps, to_document
from rewind.schema import validate
from rewind.video_io import ffprobe_meta, sample_times
from rewind.vocab import load_vocab
from tests.conftest import DEFAULT_CONFIG

from .test_run_sh import make_video

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)

CFG = load_config(DEFAULT_CONFIG)
D = 16


class FakeClip:
    """Image embedding from mean colour per frame quadrant; text from a crc32 seed."""

    logit_scale = 100.0

    def encode_images(self, rgb_frames):
        out = []
        for f in rgb_frames:
            h, w = f.shape[0] // 2, f.shape[1] // 2
            quads = [f[:h, :w], f[:h, w:], f[h:, :w], f[h:, w:]]
            v = np.concatenate([q.reshape(-1, 3).mean(axis=0) for q in quads] + [np.ones(4)])
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, dtype=np.float32)

    def encode_texts(self, texts):
        rows = [np.random.default_rng(zlib.crc32(t.encode())).normal(size=D) for t in texts]
        return np.asarray([r / np.linalg.norm(r) for r in rows], dtype=np.float32)


class NoModel(FakeClip):
    def encode_images(self, rgb_frames):
        raise AssertionError("a cached video must not be decoded again")


def models(cache: Path, clip=None) -> Models:
    return Models(
        signals=SignalModels(
            clip=clip or FakeClip(),
            new_hands=lambda: lambda rgb, t: [0.5],
            clip_version="fake-clip-v1",
            hands_version="fake-hands-v1",
        ),
        cache_root=cache,
    )


@pytest.fixture(scope="module")
def videos(tmp_path_factory) -> dict[int, Path]:
    d = tmp_path_factory.mktemp("analyze_videos")
    return {s: make_video(d / f"clip{s}.mp4", s) for s in (30, 12)}


def run(video: Path, video_id: str, m: Models) -> dict:
    result = analyze_video(video, ffprobe_meta(video), video_id, CFG, m)
    return json.loads(dumps(to_document(result)))


def test_analyze_video_30s_writes_strictly_valid_json_and_caches_signals(videos, tmp_path):
    doc = run(videos[30], "a", models(tmp_path / "cache"))

    errors, _ = validate(doc, load_vocab(), filename_stem="a", strict=True)
    assert errors == []
    for feat in ("flow", "hands", "clip"):
        assert len(list((tmp_path / "cache" / feat).glob("*.npz"))) == 1


def test_analyze_video_12s_gives_one_highlight(videos, tmp_path):
    # 12 s < SHORT_VIDEO_T (27 s): exactly one 4.05 s window (post.short_video_mode).
    doc = run(videos[12], "b", models(tmp_path / "cache"))

    errors, _ = validate(doc, load_vocab(), filename_stem="b", strict=True)
    assert errors == []
    assert len(doc["highlights"]) == 1
    s, e = doc["highlights"][0]["segment"]
    assert e - s == pytest.approx(4.05, abs=1e-9)


def test_analyze_video_renamed_copy_hits_cache_and_matches_except_video_id(videos, tmp_path):
    # The cache is keyed by content, so a renamed copy needs no decoding model, and the
    # JSON must be identical apart from video_id (CLAUDE.md §6.1).
    cache = tmp_path / "cache"
    first = run(videos[30], "a", models(cache))
    renamed = tmp_path / "Ski_totally_not_cooking.mp4"
    shutil.copy(videos[30], renamed)
    second = run(renamed, "zz", models(cache, clip=NoModel()))

    assert second["video_id"] == "zz"
    assert {**second, "video_id": "a"} == first


class NoText(NoModel):
    def encode_texts(self, texts):
        raise AssertionError("cached prompts must not be encoded again")


def test_analyze_video_new_process_reuses_cached_text_embeddings(videos, tmp_path):
    # run.sh is one process per video; the second video must not re-encode the ~450
    # prompts (~30 s on CPU). A fresh Models whose encoder can't encode anything still
    # gives the same JSON for a cached video.
    cache = tmp_path / "cache"
    first = run(videos[30], "a", models(cache))
    second = run(videos[30], "a", models(cache, clip=NoText()))

    assert second == first


def test_analyze_video_read_only_cache_still_analyzes(videos, tmp_path):
    # Regression (review of #21): an unwritable cache (read-only mount, full disk) made
    # FeatureCache.put raise, so every video fell back. The cache is only a speed-up.
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        doc = run(videos[12], "b", models(ro / "cache"))
    finally:
        ro.chmod(0o755)

    assert doc == run(videos[12], "b", models(None))


def test_text_cache_follows_label_class_order(videos, tmp_path):
    # Regression (review of #21): the text-cache key sorted label.classes, so a config
    # with the same classes in another order reused rows in the wrong order.
    from dataclasses import replace

    cache = tmp_path / "cache"
    m = models(cache)
    m.text(CFG)
    order = dict(reversed(list(CFG.label.classes.items())))
    cfg2 = replace(CFG, label=replace(CFG.label, classes=order))

    fresh = models(None).text(cfg2)
    cached = models(cache).text(cfg2)

    assert cached.label_keys == fresh.label_keys
    assert np.array_equal(cached.label, fresh.label)


def test_compute_single_pass_clip_uses_every_rth_dense_sample(videos):
    # dense 4 fps, clip 1 fps -> r = 4: clip t = 0, 1, 2, ... exactly as a 1 fps pass.
    video = videos[12]
    duration = ffprobe_meta(video).duration_sec
    p = ExtractParams.from_config(CFG)
    both = compute_single_pass(video, duration, ["flow", "clip"], FakeClip(), None, p)
    alone = compute_single_pass(video, duration, ["clip"], FakeClip(), None, p)

    assert np.array_equal(both["clip"]["t"], sample_times(duration, 1.0))
    assert np.array_equal(both["clip"]["clip_emb"], alone["clip"]["clip_emb"])
    assert np.array_equal(both["flow"]["t"], sample_times(duration, 4.0))


def test_compute_single_pass_flow_does_not_depend_on_other_missing_features(videos):
    # Regression (review of #21): frames were decoded at the widest *missing* feature's
    # width, so flow computed alone (160 px) differed from flow computed with CLIP
    # (448 px, then resized) by up to 3.5 z-units: same MP4, different JSON.
    video = videos[12]
    duration = ffprobe_meta(video).duration_sec
    p = ExtractParams.from_config(CFG)
    alone = compute_single_pass(video, duration, ["flow"], None, None, p)
    both = compute_single_pass(video, duration, ["flow", "clip"], FakeClip(), None, p)

    assert np.array_equal(alone["flow"]["flow"], both["flow"]["flow"])
