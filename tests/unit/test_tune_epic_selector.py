"""tools/tune_epic_selector.py: config variants and the adoption rule."""

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "tune_epic_selector.py"
_spec = importlib.util.spec_from_file_location("tune_epic_selector", _PATH)
tool = importlib.util.module_from_spec(_spec)
sys.modules["tune_epic_selector"] = tool
_spec.loader.exec_module(tool)


def test_with_overrides_sets_dotted_keys_without_touching_the_base():
    base = {"cluster": {"tiou": 0.3, "sim": 0.85}, "select": {"stride": 0.5}}
    out = tool.with_overrides(base, {"cluster.tiou": 0.1, "select.stride": 1.0})
    assert out == {"cluster": {"tiou": 0.1, "sim": 0.85}, "select": {"stride": 1.0}}
    assert base["cluster"]["tiou"] == 0.3


def test_with_overrides_unknown_key_is_an_error():
    with pytest.raises(KeyError):
        tool.with_overrides({"cluster": {"tiou": 0.3}}, {"cluster.tiuo": 0.1})


def test_adopt_needs_mean_gain_and_a_majority_of_videos():
    # base per video: a .5, b .5, c .5. Candidate: a .6, b .6, c .4 -> mean +0.0667,
    # better on 2 of 3 -> adopt with min_gain 0.005.
    base = {"a": 0.5, "b": 0.5, "c": 0.5}
    assert tool.adopt({"a": 0.6, "b": 0.6, "c": 0.4}, base, min_gain=0.005)
    # One big win, two losses: mean +0.0333 but better on 1 of 3 -> reject.
    assert not tool.adopt({"a": 0.8, "b": 0.45, "c": 0.45}, base, min_gain=0.005)
    # Majority better but mean gain 0.002 < 0.005 -> reject.
    assert not tool.adopt({"a": 0.503, "b": 0.503, "c": 0.5}, base, min_gain=0.005)


def test_adopt_ignores_videos_without_a_score():
    # c has no GT actions (None) -> judged on a, b only: +0.1 each -> adopt.
    assert tool.adopt({"a": 0.6, "b": 0.6, "c": None}, {"a": 0.5, "b": 0.5, "c": None}, 0.005)
