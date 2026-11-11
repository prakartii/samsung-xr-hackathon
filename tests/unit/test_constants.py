"""Kit constants: derived values and their consequences on the real manifest."""

import csv

import pytest

from rewind.constants import (
    MIN_SEG,
    MIN_SEG_EMIT,
    SCORING_BUDGET,
    SHORT_VIDEO_T,
    TIME_DECIMALS,
)
from tests.conftest import KIT_DIR


def test_short_video_threshold_is_min_seg_over_budget():
    # One 4.0 s highlight fits the 15% budget only if T >= 4.0 / 0.15 = 26.666... s.
    assert pytest.approx(26.6667, abs=1e-4) == SHORT_VIDEO_T


def test_min_seg_emit_survives_rounding_above_kit_minimum():
    # 4.05 rounded down to 2 decimals is still 4.05 >= 4.0; a window emitted at this
    # length can never be ignored for being under 4 s.
    rounded_down = int(MIN_SEG_EMIT * 10**TIME_DECIMALS) / 10**TIME_DECIMALS
    assert rounded_down >= MIN_SEG


def test_short_video_threshold_matches_three_manifest_videos():
    # ARCHITECTURE.md §5.6: exactly 3 EPIC videos (11.13, 15.28, 25.16 s) are short.
    with open(KIT_DIR / "manifest.csv", encoding="utf-8", newline="") as f:
        durations = sorted(float(r["duration_sec"]) for r in csv.DictReader(f))

    short = [d for d in durations if d < SHORT_VIDEO_T]

    assert short == pytest.approx([11.13, 15.28, 25.16], abs=1e-6)


def test_budget_is_fifteen_percent():
    assert pytest.approx(0.15, abs=1e-12) == SCORING_BUDGET
