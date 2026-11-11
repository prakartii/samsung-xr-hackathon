"""Kit constants: derived values and their consequences on the real manifest."""

import csv

import pytest

from rewind.constants import (
    MIN_SEG,
    MIN_SEG_EMIT,
    SHORT_VIDEO_T,
    TIME_DECIMALS,
)
from tests.conftest import KIT_DIR


def test_short_video_threshold_is_min_seg_over_budget():
    # One 4.0 s highlight fits the 15% budget only if T >= 4.0 / 0.15 = 26.666... s.
    assert pytest.approx(26.6667, abs=1e-4) == SHORT_VIDEO_T


def test_min_seg_emit_survives_endpoint_rounding_above_kit_minimum():
    # Post-processing rounds start up and end down to 2 decimals (ARCHITECTURE.md §5.6),
    # which can shorten a window by up to 2 * 0.01 = 0.02 s.
    # 4.05 - 0.02 = 4.03 >= 4.0 holds; with no margin, 4.0 - 0.02 = 3.98 would fail.
    worst_case = MIN_SEG_EMIT - 2 * 10**-TIME_DECIMALS
    assert worst_case >= MIN_SEG


def test_short_video_threshold_matches_three_manifest_videos():
    # ARCHITECTURE.md §5.6: exactly 3 EPIC videos (11.13, 15.28, 25.16 s) are short.
    with open(KIT_DIR / "manifest.csv", encoding="utf-8", newline="") as f:
        durations = sorted(float(r["duration_sec"]) for r in csv.DictReader(f))

    short = [d for d in durations if d < SHORT_VIDEO_T]

    assert short == pytest.approx([11.13, 15.28, 25.16], abs=1e-6)
