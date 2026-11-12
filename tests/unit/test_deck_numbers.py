"""tools/deck_numbers.py: the deck's three numbers from outputs and results files."""

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "deck_numbers.py"
_spec = importlib.util.spec_from_file_location("deck_numbers", _PATH)
tool = importlib.util.module_from_spec(_spec)
sys.modules["deck_numbers"] = tool
_spec.loader.exec_module(tool)


def test_tokens_pruned_is_one_minus_kept_share_pooled_over_videos():
    # video a: 100 s, highlights 4 + 6 = 10 s; video b: 300 s, highlights 45 s.
    # kept = (10 + 45) / (100 + 300) = 55 / 400 = 0.1375 -> pruned 0.8625
    docs = [
        {"duration_sec": 100.0, "highlights": [{"segment": [0, 4]}, {"segment": [50, 56]}]},
        {"duration_sec": 300.0, "highlights": [{"segment": [10, 55]}]},
    ]
    assert tool.tokens_pruned(docs) == pytest.approx(0.8625, abs=1e-12)


def test_sec_per_video_minute_pools_latency_rows():
    # (60 s video, 30 s) and (120 s, 60 s): 90 s / 3 min = 30 s per minute
    rows = [
        {"duration_sec": "60", "seconds": "30"},
        {"duration_sec": "120", "seconds": "60"},
    ]
    assert tool.sec_per_video_minute(rows) == pytest.approx(30.0, abs=1e-12)
