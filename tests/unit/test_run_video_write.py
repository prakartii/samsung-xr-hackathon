"""run_video writes valid JSON on any OS, without the ffmpeg CLI (PyAV-made video)."""

import json

from fixtures.videos import write_index_video

from rewind.config import load_config
from rewind.pipeline import run_video
from tests.conftest import DEFAULT_CONFIG


def test_run_video_stage_failure_still_writes_fallback_json(tmp_path):
    # 30 frames at 1 fps = 30.0 s; a failing stage leaves the fallback on disk:
    # no actions, and the duration echoed from the probe (30.0 s, within 0.11 s).
    video = write_index_video(tmp_path / "clip.avi", n_frames=30, fps=1.0)

    def broken(*_):
        raise RuntimeError("stage failed")

    out = tmp_path / "out" / "clip.json"
    run_video(video, out, load_config(DEFAULT_CONFIG), team_id="T07", analyze=broken)

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["actions"] == []
    assert abs(doc["duration_sec"] - 30.0) <= 0.11
    assert [p.name for p in out.parent.iterdir()] == ["clip.json"]  # no temp file left
