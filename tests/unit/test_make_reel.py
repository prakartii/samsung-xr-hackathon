"""tools/make_reel.py: clip plan, captions and fades for the demo reel (P8)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "make_reel.py"
_spec = importlib.util.spec_from_file_location("make_reel", _PATH)
tool = importlib.util.module_from_spec(_spec)
sys.modules["make_reel"] = tool
_spec.loader.exec_module(tool)


def test_reel_plan_pads_clips_to_the_video_and_merges_overlaps():
    # T = 60, pad 1: [0.5, 5] -> [0, 6]; [6.5, 10] -> [5.5, 11] overlaps [0, 6] -> [0, 11];
    # [57, 59.8] -> [56, 60].
    hl = [(0.5, 5.0), (6.5, 10.0), (57.0, 59.8)]
    assert tool.reel_plan(hl, 60.0, pad=1.0) == [(0.0, 11.0), (56.0, 60.0)]


def test_reel_plan_without_padding_keeps_highlights():
    assert tool.reel_plan([(10.0, 14.0), (20.0, 24.0)], 60.0, pad=0.0) == [
        (10.0, 14.0),
        (20.0, 24.0),
    ]


def test_captions_at_lists_active_actions_by_score_without_repeats():
    # t = 5: active are take plate (0.9), wash plate (0.4), take plate again (0.5),
    # open tap (0.7); [6, 8] is not active. Top 2 distinct, by score: take plate, open tap.
    actions = [
        {"verb": "take", "noun": "plate", "segment": [4.0, 6.0], "score": 0.9},
        {"verb": "wash", "noun": "plate", "segment": [0.0, 9.0], "score": 0.4},
        {"verb": "take", "noun": "plate", "segment": [3.0, 5.5], "score": 0.5},
        {"verb": "open", "noun": "tap", "segment": [5.0, 7.0], "score": 0.7},
        {"verb": "close", "noun": "tap", "segment": [6.0, 8.0], "score": 0.99},
    ]
    assert tool.captions_at(5.0, actions, top_k=2) == ["take plate", "open tap"]


def test_captions_at_nothing_active_is_empty():
    assert (
        tool.captions_at(1.0, [{"verb": "a", "noun": "b", "segment": [2, 3], "score": 1}], 2) == []
    )


@pytest.mark.parametrize(
    ("t", "expect"),
    [
        (0.0, 0.0),  # clip start: black
        (0.15, 0.5),  # halfway through a 0.3 s fade-in
        (2.0, 1.0),  # middle of a 4 s clip
        (3.85, 0.5),  # 0.15 s before the end: halfway through the fade-out
    ],
)
def test_fade_gain_ramps_in_and_out(t, expect):
    assert tool.fade_gain(t, clip_len=4.0, fade=0.3) == pytest.approx(expect, abs=1e-9)


def test_fade_gain_zero_fade_is_always_full():
    assert tool.fade_gain(0.0, clip_len=4.0, fade=0.0) == pytest.approx(1.0, abs=1e-12)
