"""Cross-validated config choice for FPVSum tuning (tools/tune_fpvsum.py)."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "tune_fpvsum.py"
_spec = importlib.util.spec_from_file_location("tune_fpvsum", _PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

# 2 configs, 4 videos, 2 folds: fold 0 = {a, b}, fold 1 = {c, d}.
# Config "x" is better on fold 0's videos, config "y" on fold 1's.
SCORES = {
    "x": {"a": 0.9, "b": 0.7, "c": 0.1, "d": 0.3},
    "y": {"a": 0.2, "b": 0.4, "c": 0.8, "d": 0.6},
}
FOLDS = {"a": 0, "b": 0, "c": 1, "d": 1}


def test_cv_picks_on_training_folds_and_scores_held_out():
    cv = tool.cross_validate(SCORES, FOLDS)
    # Held out fold 0: train on {c, d}: x = 0.2, y = 0.7 -> pick y; held-out y on {a, b} = 0.3
    # Held out fold 1: train on {a, b}: x = 0.8, y = 0.3 -> pick x; held-out x on {c, d} = 0.2
    assert cv["per_fold"] == {0: ("y", pytest.approx(0.3)), 1: ("x", pytest.approx(0.2))}
    # The honest estimate: mean of held-out scores = (0.3 + 0.2) / 2 = 0.25
    assert cv["cv_mean"] == pytest.approx(0.25, abs=1e-12)


def test_cv_final_choice_uses_all_videos():
    # All 4 videos: x = (0.9 + 0.7 + 0.1 + 0.3) / 4 = 0.5, y = (0.2 + 0.4 + 0.8 + 0.6) / 4 = 0.5
    # -> tie, broken by the first config in the given order ("x")
    assert tool.cross_validate(SCORES, FOLDS)["chosen"] == "x"


def test_cv_final_choice_prefers_higher_mean():
    scores = {"x": {"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1}, "y": dict(SCORES["y"])}
    assert tool.cross_validate(scores, FOLDS)["chosen"] == "y"
