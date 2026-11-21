"""tools/eval_yth_f1.py: YTH matched-clip intervals and time-grid highlight F1."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "eval_yth_f1.py"
_spec = importlib.util.spec_from_file_location("eval_yth_f1", _PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_positive_intervals_overlapping_matches_are_merged_and_converted_to_seconds():
    # 25 fps. Clips 1 and 2 (labels 1) overlap: [50,150] union [100,200] = [50,200] frames
    # = [2.0, 8.0] s. Clip 3 (label 0, borderline) and clip 0 (-1) are negative.
    # Clip 4 [300,400] = [12.0, 16.0] s stays separate.
    clips = [[0, 100], [50, 150], [100, 200], [150, 250], [300, 400]]
    labels = [-1, 1, 1, 0, 1]

    got = tool.positive_intervals(clips, labels, fps=25.0)

    assert len(got) == 2
    assert got[0] == pytest.approx((2.0, 8.0), abs=1e-12)
    assert got[1] == pytest.approx((12.0, 16.0), abs=1e-12)


def test_positive_intervals_touching_clips_merge():
    # 10 fps: [0,50] and [50,100] touch at frame 50 -> one interval [0, 10] s.
    got = tool.positive_intervals([[50, 100], [0, 50]], [1, 1], fps=10.0)

    assert len(got) == 1
    assert got[0] == pytest.approx((0.0, 10.0), abs=1e-12)


def test_frame_f1_known_overlap_matches_hand_count():
    # Grid step 0.5 over [0, 20): t = 0, 0.5, ..., 19.5 (40 samples).
    # pred [10, 14.05) (4.05 s >= 4): t = 10.0 .. 14.0 -> 9 samples.
    # gt   [12, 16):                  t = 12.0 .. 15.5 -> 8 samples.
    # TP = t in [12, 14.05) = 12.0 .. 14.0 -> 5. FP = 9 - 5 = 4. FN = 8 - 5 = 3.
    # P = 5/9, R = 5/8, F1 = 2·TP / (2·TP + FP + FN) = 10 / 17.
    f1 = tool.frame_f1([(10.0, 14.05)], [(12.0, 16.0)], duration=20.0, step=0.5)

    assert f1 == pytest.approx(10 / 17, abs=1e-9)


def test_frame_f1_highlight_under_4s_is_ignored():
    # pred [12, 15.9) is 3.9 s < 4.0 -> no predicted samples, even though it sits
    # entirely inside gt [12, 16) -> TP = 0 -> F1 = 0.
    f1 = tool.frame_f1([(12.0, 15.9)], [(12.0, 16.0)], duration=20.0, step=0.5)

    assert f1 == pytest.approx(0.0, abs=1e-12)


def test_frame_f1_disjoint_prediction_is_zero():
    # pred [0, 4.05) and gt [12, 16) share no sample -> TP = 0 -> F1 = 0.
    f1 = tool.frame_f1([(0.0, 4.05)], [(12.0, 16.0)], duration=20.0, step=0.5)

    assert f1 == pytest.approx(0.0, abs=1e-12)


def test_frame_f1_exact_prediction_is_one():
    # pred == gt == [4, 8) (4.0 s, just at the minimum): 8 samples each, all TP.
    f1 = tool.frame_f1([(4.0, 8.0)], [(4.0, 8.0)], duration=20.0, step=0.5)

    assert f1 == pytest.approx(1.0, abs=1e-12)


def test_span_matches_flags_truncated_download():
    # Labels end at frame 5644 at 29.97 fps = 188.3 s; a 47.1 s file (partial
    # download) is off by 141 s > max(2, 0.05·47.1) -> mismatch. A 188.0 s file
    # is off by 0.3 s -> match.
    assert not tool.span_matches(5644, 29.97, 47.1)
    assert tool.span_matches(5644, 29.97, 188.0)
