"""Alignment arithmetic of tools/import_released_features.py."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "import_released_features.py"
_spec = importlib.util.spec_from_file_location("import_released_features", _PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_expected_feature_count_hand_computed():
    # 10 s -> 300 frames at 30 fps -> (300 - 32) // 16 + 1 = 268 // 16 + 1 = 16 + 1 = 17
    assert tool.expected_feature_count(10.0) == 17


def test_expected_feature_count_shorter_than_one_clip_is_zero():
    # 1 s = 30 frames < 32 -> no full clip
    assert tool.expected_feature_count(1.0) == 0


def test_alignment_gap_in_seconds():
    # 10 s expects 17 features; 20 features -> 3 steps * 16/30 s = 1.6 s
    assert tool.alignment_gap_sec(20, 10.0) == pytest.approx(1.6, abs=1e-12)
    assert tool.alignment_gap_sec(17, 10.0) == pytest.approx(0.0, abs=1e-12)
