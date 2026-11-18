"""tools/tune_label.py: setting choice and nested CV."""

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
_spec = importlib.util.spec_from_file_location("tune_label", _TOOLS / "tune_label.py")
tool = importlib.util.module_from_spec(_spec)
sys.modules["tune_label"] = tool
_spec.loader.exec_module(tool)


def test_choose_prefers_current_on_a_tie():
    acc = {(8, "prob"): 0.8, (16, "prob"): 0.8, (32, "logprob"): 0.7}
    assert tool.choose(acc, (8, "prob")) == (8, "prob")


def test_choose_tie_without_current_takes_fewer_frames_all_last():
    acc = {(0, "prob"): 0.9, (32, "prob"): 0.9, (8, "prob"): 0.5}
    assert tool.choose(acc, (8, "prob")) == (32, "prob")


def test_nested_cv_scores_each_fold_with_the_setting_chosen_on_the_others():
    # folds: a, b -> 0; c, d -> 1. Setting X right on a, b, c; Y right on c, d only.
    # fold 0 held out: train c, d -> X 1/2, Y 2/2 -> Y; Y on a, b: 0 hits.
    # fold 1 held out: train a, b -> X 2/2, Y 0 -> X; X on c, d: 1 hit.
    # selection 1/4; current X on all four: 3/4.
    X, Y = (8, "prob"), (16, "prob")
    correct = {X: {"a": True, "b": True, "c": True, "d": False},
               Y: {"a": False, "b": False, "c": True, "d": True}}  # fmt: skip
    sel, cur, chosen = tool.nested_cv(correct, {"a": 0, "b": 0, "c": 1, "d": 1}, X)
    assert sel == pytest.approx(0.25) and cur == pytest.approx(0.75)
    assert chosen == [Y, X]
