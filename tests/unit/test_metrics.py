"""Scorer metrics on tiny hand-computed cases (CLAUDE.md §7.2). Arithmetic in comments."""

import numpy as np
import pytest

from rewind_eval.coverage import budget_penalty, coverage, mean_ignoring_none
from rewind_eval.f1 import frame_f1, top_frames
from rewind_eval.label_acc import label_accuracy
from rewind_eval.map import class_ap, interpolated_ap, mean_ap, tiou

# --- coverage ------------------------------------------------------------------
# T = 100 s. GT: 4 instances, 3 distinct pairs (cut onion appears twice).
GT = [
    ((10.0, 12.0), "open", "fridge"),
    ((50.0, 55.0), "cut", "onion"),
    ((80.0, 81.0), "cut", "onion"),
    ((90.0, 95.0), "wash", "knife"),
]


def cov(highlights, rule="any_overlap", **kw):
    return coverage(GT, highlights, 100.0, rule=rule, **kw)


def test_coverage_one_of_three_pairs_captured():
    # (9, 13.05) overlaps open/fridge only → 1/3.
    assert cov([(9.0, 13.05)]) == pytest.approx(1 / 3, abs=1e-12)


def test_coverage_repeated_pair_counts_once():
    # Both cut/onion instances captured, still one pair → 1/3.
    assert cov([(49.0, 53.05), (79.0, 83.05)]) == pytest.approx(1 / 3, abs=1e-12)


def test_coverage_any_overlap_captures_small_overlap():
    # (54, 58.05) overlaps (50, 55) by 1 s → captured under any_overlap.
    assert cov([(54.0, 58.05)]) == pytest.approx(1 / 3, abs=1e-12)


def test_coverage_fraction_rule_needs_half_the_span():
    # Same 1 s of a 5 s span: 1 / 5 = 0.2 < 0.5 → not captured.
    assert cov([(54.0, 58.05)], rule="fraction", theta=0.5) == pytest.approx(0.0, abs=1e-12)


def test_coverage_fraction_rule_captures_other_instance_of_same_pair():
    # (80, 81) is fully inside (79, 83.05): 1 / 1 = 1.0 ≥ 0.5 → cut/onion captured.
    assert cov([(79.0, 83.05)], rule="fraction") == pytest.approx(1 / 3, abs=1e-12)


def test_coverage_highlight_under_4s_is_ignored():
    # (9, 12.99) is 3.99 s: ignored for capture (kit §4).
    assert cov([(9.0, 12.99)]) == pytest.approx(0.0, abs=1e-12)


def test_coverage_touching_highlight_does_not_capture():
    # (5, 10) ends exactly where open/fridge starts: zero overlap.
    assert cov([(5.0, 10.0)]) == pytest.approx(0.0, abs=1e-12)


def test_coverage_over_budget_is_penalized():
    # All 3 pairs captured, total 5 + 5 + 5 + 5 = 20 s = 0.20 · T.
    # over = 0.20 - 0.15 = 0.05; penalty = 1 - 1.0 · 0.05 / 0.15 = 2/3 → 1 · 2/3.
    hl = [(8.0, 13.0), (50.0, 55.0), (77.0, 82.0), (90.0, 95.0)]

    assert cov(hl) == pytest.approx(2 / 3, abs=1e-12)


def test_coverage_penalty_slope_zero_disables_penalty():
    hl = [(8.0, 13.0), (50.0, 55.0), (77.0, 82.0), (90.0, 95.0)]

    assert cov(hl, lam=0.0) == pytest.approx(1.0, abs=1e-12)


def test_budget_penalty_floors_at_zero():
    # 45 s of 100: over = 0.30, 1 - 0.30 / 0.15 = -1 → 0.
    assert budget_penalty([(0.0, 45.0)], 100.0, 1.0) == pytest.approx(0.0, abs=1e-12)


def test_coverage_video_without_gt_is_none_and_skipped_in_mean():
    assert coverage([], [(0.0, 5.0)], 100.0, rule="any_overlap") is None
    # mean of 0.5 and 1.0, the None excluded → 0.75
    assert mean_ignoring_none([0.5, None, 1.0]) == pytest.approx(0.75, abs=1e-12)


# --- mAP ----------------------------------------------------------------------


def test_tiou_partial_overlap():
    # (0, 10) vs (5, 15): intersection 5, union 15 → 1/3.
    assert tiou((0.0, 10.0), (5.0, 15.0)) == pytest.approx(1 / 3, abs=1e-12)


def test_interpolated_ap_with_a_false_positive_in_the_middle():
    # TP, FP, TP with 2 GT: recall .5, .5, 1; precision 1, .5, 2/3.
    # Envelope: 1, 2/3, 2/3. AP = .5·1 + 0·(2/3) + .5·(2/3) = 0.8333...
    assert interpolated_ap([True, False, True], 2) == pytest.approx(5 / 6, abs=1e-12)


def test_interpolated_ap_uses_precision_envelope():
    # FP, TP, TP with 2 GT: recall 0, .5, 1; precision 0, .5, 2/3.
    # Envelope (max precision at any higher recall): 2/3, 2/3, 2/3.
    # AP = 0·(2/3) + .5·(2/3) + .5·(2/3) = 2/3. Without it: .5·.5 + .5·(2/3) = 0.5833.
    assert interpolated_ap([False, True, True], 2) == pytest.approx(2 / 3, abs=1e-12)


def test_class_ap_duplicate_detection_after_match_is_false_positive():
    # Two identical predictions on one GT: TP then FP. recall 1, 1; precision 1, .5.
    # Envelope 1, .5 → AP = 1·1 + 0·.5 = 1.0 (a trailing duplicate doesn't lower AP).
    preds = [("v", (0.0, 10.0), 0.9), ("v", (0.0, 10.0), 0.8)]

    assert class_ap(preds, {"v": [(0.0, 10.0)]}, 0.5) == pytest.approx(1.0, abs=1e-12)


def test_class_ap_ties_keep_input_order():
    # Equal scores: the miss comes first in input order, so FP then TP.
    # recall 0, 1; precision 0, .5 → envelope .5, .5 → AP = 0 + 1·.5 = 0.5.
    preds = [("v", (50.0, 60.0), 0.5), ("v", (0.0, 10.0), 0.5)]

    assert class_ap(preds, {"v": [(0.0, 10.0)]}, 0.5) == pytest.approx(0.5, abs=1e-12)


def test_class_ap_matches_highest_tiou_gt_not_first():
    # GT in order (1, 11), (0, 10); threshold 0.8.
    # p1 (0, 10): tIoU 9/11 = .818 with (1, 11), 1.0 with (0, 10) → takes (0, 10).
    # p2 (1.2, 11.2): 9.8/10.2 = .96 with (1, 11) → TP. AP = 1.0.
    # (Taking the first eligible GT would give p1 → (1, 11) and leave p2 with
    # (0, 10) at 8.8/11.2 = .786 < .8 → FP, AP 0.5.)
    preds = [("v", (0.0, 10.0), 0.9), ("v", (1.2, 11.2), 0.8)]

    ap = class_ap(preds, {"v": [(1.0, 11.0), (0.0, 10.0)]}, 0.8)

    assert ap == pytest.approx(1.0, abs=1e-12)


def test_class_ap_prediction_in_wrong_video_does_not_match():
    preds = [("w", (0.0, 10.0), 0.9)]

    assert class_ap(preds, {"v": [(0.0, 10.0)]}, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_mean_ap_each_threshold():
    # tIoU 1/3: a hit at 0.1, 0.2, 0.3 and a miss at 0.4, 0.5 → avg 3/5 = 0.6.
    preds = {"v": [((0.0, 10.0), "cut", "onion", 0.9)]}
    gts = {"v": [((5.0, 15.0), "cut", "onion")]}

    m = mean_ap(preds, gts)

    assert [m[k] for k in ("0.1", "0.2", "0.3", "0.4", "0.5")] == pytest.approx(
        [1, 1, 1, 0, 0], abs=1e-12
    )
    assert m["avg"] == pytest.approx(0.6, abs=1e-12)


def test_mean_ap_class_without_predictions_scores_zero():
    # Two GT classes, one predicted perfectly → (1 + 0) / 2 = 0.5.
    gts = {"v": [((0.0, 10.0), "cut", "onion"), ((20.0, 30.0), "wash", "knife")]}
    preds = {"v": [((0.0, 10.0), "cut", "onion", 0.9)]}

    assert mean_ap(preds, gts)["avg"] == pytest.approx(0.5, abs=1e-12)


def test_mean_ap_perfect_is_one_and_empty_is_zero():
    gts = {"v": [((0.0, 10.0), "cut", "onion")], "w": [((5.0, 9.0), "open", "fridge")]}
    perfect = {vid: [(s, v, n, 1.0) for s, v, n in items] for vid, items in gts.items()}

    assert mean_ap(perfect, gts)["avg"] == pytest.approx(1.0, abs=1e-12)
    assert mean_ap({}, gts)["avg"] == pytest.approx(0.0, abs=1e-12)


# --- frame F1 -----------------------------------------------------------------
TIMES = np.arange(20, dtype=float)  # 20 frames at 1 fps
IMP = np.zeros(20)
IMP[[10, 11, 12]] = 1.0  # top 15% of 20 = 3 frames: 10, 11, 12


def test_frame_f1_partial_overlap():
    # (10, 14.05) → frames 10..14 = 5 predicted, 3 true positives.
    # P = 3/5, R = 3/3 → F1 = 2 · .6 · 1 / 1.6 = 0.75.
    assert frame_f1(TIMES, top_frames(IMP), [(10.0, 14.05)]) == pytest.approx(0.75, abs=1e-12)


def test_frame_f1_highlight_under_4s_is_ignored():
    assert frame_f1(TIMES, top_frames(IMP), [(10.0, 13.0)]) == pytest.approx(0.0, abs=1e-12)


def test_frame_f1_highlight_end_is_exclusive():
    # (6, 10) covers frames 6..9, not 10 → no true positives.
    assert frame_f1(TIMES, top_frames(IMP), [(6.0, 10.0)]) == pytest.approx(0.0, abs=1e-12)


def test_frame_f1_perfect_is_one():
    # top 20% of 20 = 4 frames, 10..13; (10, 14) predicts exactly those.
    imp = np.zeros(20)
    imp[[10, 11, 12, 13]] = 1.0

    assert frame_f1(TIMES, top_frames(imp, 0.2), [(10.0, 14.0)]) == pytest.approx(1.0, abs=1e-12)


def test_top_frames_count_rounds_up_and_ties_take_earlier_frames():
    # 15% of 10 = 1.5 → 2 frames; all equal → frames 0 and 1.
    mask = top_frames(np.ones(10))

    assert np.flatnonzero(mask).tolist() == [0, 1]


# --- label accuracy -----------------------------------------------------------


def test_label_accuracy_missing_prediction_counts_as_wrong():
    # a right, b wrong, c missing → 1/3.
    pred = {"a": "ski", "b": "bike"}
    gt = {"a": "ski", "b": "surfing", "c": "boxing"}

    assert label_accuracy(pred, gt) == pytest.approx(1 / 3, abs=1e-12)


# --- review fixes (PR #12) ------------------------------------------------------


def test_class_ap_ranks_predictions_by_score_descending():
    # Ranked: FP (0.9) then TP (0.1). recall 0, 1; precision 0, .5
    # → envelope .5, .5 → AP = 1 · .5 = 0.5 (ascending or unsorted gives 1.0).
    preds = [("v", (0.0, 10.0), 0.1), ("v", (50.0, 60.0), 0.9)]

    assert class_ap(preds, {"v": [(0.0, 10.0)]}, 0.5) == pytest.approx(0.5, abs=1e-12)


def test_class_ap_tiou_equal_to_threshold_is_a_match():
    # (0, 10) vs (5, 10): tIoU 5/10 = 0.5 exactly; ActivityNet counts ≥ threshold.
    preds = [("v", (0.0, 10.0), 0.9)]

    assert class_ap(preds, {"v": [(5.0, 10.0)]}, 0.5) == pytest.approx(1.0, abs=1e-12)


def test_coverage_highlight_of_exactly_4s_counts():
    # (9, 13) is exactly 4.0 s: not ignored; captures open/fridge → 1/3.
    assert cov([(9.0, 13.0)]) == pytest.approx(1 / 3, abs=1e-12)


def test_coverage_short_highlights_still_count_toward_the_budget():
    # (8, 13) 5 s captures open/fridge (1/3). Three 3.9 s highlights capture nothing but
    # add 11.7 s: total 16.7 s = 0.167·T → over = 0.017, penalty 1 − .017/.15 = .8867.
    hl = [(8.0, 13.0), (20.0, 23.9), (30.0, 33.9), (60.0, 63.9)]

    assert cov(hl) == pytest.approx(1 / 3 * (1 - 0.017 / 0.15), abs=1e-9)
