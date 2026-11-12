"""tools/latency.py: seconds of processing per minute of video (P7, deck number)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "latency.py"
_spec = importlib.util.spec_from_file_location("latency", _PATH)
tool = importlib.util.module_from_spec(_spec)
sys.modules["latency"] = tool  # dataclasses resolve the module by name
_spec.loader.exec_module(tool)
Run, summarize = tool.Run, tool.summarize


def test_summarize_pools_seconds_over_video_minutes_per_route():
    # epic: (30 + 90) s over (60 + 120) s of video = 120 s / 3 min = 40 s per minute
    # fpvsum: 10 s over 120 s = 10 s / 2 min = 5 s per minute
    # all: 130 s / 5 min = 26 s per minute
    runs = [
        Run("a", "epic", 60.0, 30.0),
        Run("b", "epic", 120.0, 90.0),
        Run("c", "fpvsum", 120.0, 10.0),
    ]
    out = summarize(runs)
    assert out["epic"]["sec_per_video_min"] == pytest.approx(40.0, abs=1e-9)
    assert out["fpvsum"]["sec_per_video_min"] == pytest.approx(5.0, abs=1e-9)
    assert out["all"]["sec_per_video_min"] == pytest.approx(26.0, abs=1e-9)
    assert (out["epic"]["videos"], out["all"]["videos"]) == (2, 3)


def test_summarize_reports_slowest_video_per_minute():
    # a: 30 s / 1 min = 30; b: 90 s / 2 min = 45 -> worst 45 s per minute
    out = summarize([Run("a", "epic", 60.0, 30.0), Run("b", "epic", 120.0, 90.0)])
    assert out["epic"]["worst_sec_per_video_min"] == pytest.approx(45.0, abs=1e-9)
