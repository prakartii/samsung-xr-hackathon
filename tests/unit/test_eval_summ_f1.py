"""tools/eval_summ_f1.py: TVSum/SumMe importance -> FPVSum-style top-15% GT and F1."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "eval_summ_f1.py"
_spec = importlib.util.spec_from_file_location("eval_summ_f1", _PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_tvsum_scores_mean_over_annotators_per_video():
    # v1: annotators [1,3,5] and [3,3,1] -> mean [2,3,3]. v2 has one annotator.
    rows = [("v1", "BK", "1,3,5"), ("v2", "VT", "4,2"), ("v1", "BK", "3,3,1")]

    got = tool.tvsum_scores(rows)

    assert got["v1"] == pytest.approx([2.0, 3.0, 3.0], abs=1e-12)
    assert got["v2"] == pytest.approx([4.0, 2.0], abs=1e-12)


def test_tvsum_scores_unequal_frame_counts_raise():
    with pytest.raises(ValueError):
        tool.tvsum_scores([("v", "c", "1,2"), ("v", "c", "1,2,3")])


def test_summe_scores_fraction_of_users_selecting_frame():
    # 4 users x 3 frames: column sums 4, 1, 0 -> 1.0, 0.25, 0.0.
    us = np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])

    assert tool.summe_scores(us) == pytest.approx([1.0, 0.25, 0.0], abs=1e-12)


def test_gt_mask_top_15_percent_frames_at_uniform_times():
    # 20 frames over 10 s -> t_i = 0.5 i. k = ceil(0.15 * 20) = 3 top frames.
    # Scores: frames 7, 8, 9 highest (9, 8, 7); others 0 -> mask exactly {7, 8, 9}.
    score = np.zeros(20)
    score[[7, 8, 9]] = [9.0, 8.0, 7.0]

    times, mask = tool.gt_mask(score, 10.0)

    assert times[1] == pytest.approx(0.5, abs=1e-12)
    assert times[19] == pytest.approx(9.5, abs=1e-12)
    assert list(np.flatnonzero(mask)) == [7, 8, 9]


def test_score_video_known_overlap_matches_hand_count():
    # 40 frames over 20 s -> t_i = 0.5 i. k = ceil(0.15 * 40) = 6 GT frames: 20..25
    # (t = 10.0 .. 12.5). Highlight [11, 15.05) (>= 4 s) covers t = 11.0 .. 15.0 ->
    # frames 22..30 = 9 predicted. TP = frames 22..25 = 4.
    # P = 4/9, R = 4/6 -> F1 = 2*(4/9)(4/6)/((4/9)+(4/6)) = (16/27)/(10/9) = 8/15.
    score = np.zeros(40)
    score[20:26] = 1.0

    got = tool.score_video(score, 20.0, [(11.0, 15.05)])

    assert got == pytest.approx(8 / 15, abs=1e-12)


def test_score_video_highlight_under_4s_counts_nothing():
    # Same GT; a 3.9 s highlight over all GT frames is ignored -> no TP -> 0.
    score = np.zeros(40)
    score[20:26] = 1.0

    assert tool.score_video(score, 20.0, [(10.0, 13.9)]) == pytest.approx(0.0, abs=1e-12)
