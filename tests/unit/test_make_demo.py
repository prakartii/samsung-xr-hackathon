"""tools/make_demo.py: clip choice, layout and results parsing (pure parts)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(_TOOLS))
_spec = importlib.util.spec_from_file_location("make_demo", _TOOLS / "make_demo.py")
tool = importlib.util.module_from_spec(_spec)
sys.modules["make_demo"] = tool
_spec.loader.exec_module(tool)


def test_pick_evenly_keeps_first_and_last():
    # 9 items, 3 picks -> indices 0, 4, 8
    assert tool.pick_evenly(list(range(9)), 3) == [0, 4, 8]


def test_pick_evenly_fewer_items_than_asked_returns_all():
    assert tool.pick_evenly([1, 2], 5) == [1, 2]


def test_demo_clips_cuts_each_clip_to_max_len():
    # three highlights, keep 2 evenly (first and last), each cut to 4.5 s
    hls = [(10.0, 20.0), (30.0, 34.0), (50.0, 58.0)]
    assert tool.demo_clips(hls, 100.0, 2, 4.5) == [(10.0, 14.5), (50.0, 54.5)]


def test_demo_clips_merges_touching_highlights_first():
    # (0, 4) and (4, 8) touch -> one clip (0, 8), cut to 5 s
    assert tool.demo_clips([(0.0, 4.0), (4.0, 8.0)], 20.0, 3, 5.0) == [(0.0, 5.0)]


def test_fit_box_wide_source_fills_the_width():
    # 1920x1080 into 1280x544: height-bound -> 544/1080 = 0.5037 -> 967x544 -> even 966x544
    assert tool.fit_box(1920, 1080, 1280, 544) == (966, 544)


def test_fit_box_tall_source_fills_the_height():
    # 480x640 into 1280x544: scale 0.85 -> 408x544
    assert tool.fit_box(480, 640, 1280, 544) == (408, 544)


def test_sample_times_even_spacing_from_start():
    assert tool.sample_times(10.0, 12.0, 4) == pytest.approx([10.0, 10.5, 11.0, 11.5])


def test_parse_ablation_reads_rows_and_skips_blank_cells():
    md = (
        "Ablation on `report`.\n\n"
        "| variant | weighted | map |\n|---|---|---|\n"
        "| full pipeline | 0.3614 | 0.1493 |\n| uniform baseline | 0.1763 |  |\n"
    )
    out = tool.parse_ablation(md)
    assert out["full pipeline"] == {"weighted": pytest.approx(0.3614), "map": pytest.approx(0.1493)}
    assert out["uniform baseline"] == {"weighted": pytest.approx(0.1763)}


def test_clock_formats_minutes_and_seconds():
    assert tool.clock(562.4) == "9:22"
    assert tool.clock(5.0) == "0:05"
