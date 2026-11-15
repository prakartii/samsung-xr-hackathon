"""tools/run_all.py: which manifest videos get run, skipped or reported missing."""

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "tools" / "run_all.py"
_spec = importlib.util.spec_from_file_location("run_all", _PATH)
tool = importlib.util.module_from_spec(_spec)
sys.modules["run_all"] = tool
_spec.loader.exec_module(tool)


def test_plan_runs_present_videos_skips_done_ones_and_lists_missing(tmp_path):
    # a: MP4 present, no output -> run. b: MP4 and output present, skip_existing ->
    # skipped. c: no MP4 -> missing.
    videos, out = tmp_path / "videos", tmp_path / "out"
    videos.mkdir()
    out.mkdir()
    for v in ("a", "b"):
        (videos / f"{v}.mp4").write_bytes(b"x")
    (out / "b.json").write_text("{}", encoding="utf-8")

    jobs, skipped, missing = tool.plan(["a", "b", "c"], videos, out, skip_existing=True)

    assert jobs == [(videos / "a.mp4", out / "a.json")]
    assert skipped == ["b"]
    assert missing == ["c"]


def test_plan_without_skip_reruns_existing_outputs(tmp_path):
    videos, out = tmp_path / "videos", tmp_path / "out"
    videos.mkdir()
    out.mkdir()
    (videos / "b.mp4").write_bytes(b"x")
    (out / "b.json").write_text("{}", encoding="utf-8")

    jobs, skipped, _ = tool.plan(["b"], videos, out, skip_existing=False)

    assert jobs == [(videos / "b.mp4", out / "b.json")] and skipped == []
