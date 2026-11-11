"""Baselines, oracle and compute_scores on hand-computed cases."""

import itertools

import numpy as np
import pytest

from rewind_eval.baselines import random_highlights, uniform_highlights
from rewind_eval.coverage import coverage
from rewind_eval.gt_fpvsum import FpvGT
from rewind_eval.oracle import coverage_oracle, f1_oracle
from rewind_eval.score import WEIGHTS, EvalData, Prediction, compute_scores

# --- baselines ---------------------------------------------------------------


def test_uniform_highlights_three_windows_rounded_inward():
    # B = 0.145 · 100 = 14.5 → floor(14.5 / 4.05) = 3 windows, centres 16.67, 50, 83.33:
    # (14.6417, 18.6917) → (14.65, 18.69); (47.975, 52.025) → (47.98, 52.02);
    # (81.3083, 85.3583) → (81.31, 85.35).
    assert uniform_highlights(100.0, 0.145) == [(14.65, 18.69), (47.98, 52.02), (81.31, 85.35)]


def test_uniform_highlights_short_video_gets_one_window():
    # T = 12 < 27: one window at 6 ± 2.025 → (3.98, 8.02).
    assert uniform_highlights(12.0, 0.145) == [(3.98, 8.02)]


def test_random_highlights_seeded_disjoint_and_counted():
    a = random_highlights(300.0, 0.145, seed=3)

    assert a == random_highlights(300.0, 0.145, seed=3)
    assert len(a) == 10  # floor(0.145 · 300 / 4.05) = floor(10.74) = 10
    assert all(e - s >= 4.0 for s, e in a) and all(s >= 0 and e <= 300 for s, e in a)
    assert all(e0 < s1 for (_, e0), (s1, _) in itertools.pairwise(a))


# --- oracle ------------------------------------------------------------------
GT = [((10.0, 11.0), "a", "x"), ((12.0, 13.0), "b", "x"), ((50.0, 51.0), "c", "x")]


def test_coverage_oracle_first_pick_captures_two_pairs():
    # Budget 0.06 · 100 = 6 s: one 4.05 s window. Windows overlapping both (10, 11) and
    # (12, 13) start in (7.95, 11); the first on the 0.5 s grid is 8.0 → (8.0, 12.05).
    assert coverage_oracle(GT, 100.0, budget_frac=0.06) == [(8.0, 12.05)]


def test_coverage_oracle_with_room_captures_everything():
    hl = coverage_oracle(GT, 100.0, budget_frac=0.145)

    assert coverage(GT, hl, 100.0, rule="any_overlap") == pytest.approx(1.0, abs=1e-12)
    assert sum(e - s for s, e in hl) <= 14.5 + 1e-9


def test_coverage_oracle_short_video_gets_one_window_despite_budget():
    # T = 20 < 27: budget 2.9 s < 4.05, but one 4.05 s window is still allowed.
    hl = coverage_oracle([((5.0, 6.0), "a", "x")], 20.0, budget_frac=0.145)

    assert len(hl) == 1
    assert hl[0][1] - hl[0][0] == pytest.approx(4.05, abs=1e-9)


def test_f1_oracle_covers_a_ten_frame_highlight_with_two_windows():
    # Frames at 0..99 s, GT positives 40..49. First pick (40, 44.05) covers 40..44 (5).
    # Next window must start ≥ 44.06: (45.0, 49.05) covers 45..49 (5). Then nothing left.
    times = np.arange(100, dtype=float)
    gt = (times >= 40) & (times < 50)

    assert f1_oracle(times, gt, 100.0, budget_frac=0.145) == [(40.0, 44.05), (45.0, 49.05)]


def test_f1_oracle_window_end_is_exclusive():
    # GT frames 44 and 48; 4 s windows on a 1 s grid. [44, 48) holds 44 but not 48, so
    # no window covers both: first pick (41, 45) (earliest with gain 1), then the next
    # start ≥ 45.01 is 46 → (46, 50) for frame 48. (An inclusive end would take (44, 48).)
    times = np.arange(100, dtype=float)
    gt = np.isin(times, [44.0, 48.0])

    hl = f1_oracle(times, gt, 100.0, budget_frac=0.145, lengths=(4.0,), stride=1.0)

    assert hl == [(41.0, 45.0), (46.0, 50.0)]


# --- compute_scores ----------------------------------------------------------


def make_data():
    times = np.arange(20, dtype=float)
    hl = np.zeros(20, dtype=bool)
    hl[[10, 11, 12]] = True
    return EvalData(
        epic_gt={"E": [((10.0, 11.0), "cut", "onion"), ((50.0, 51.0), "wash", "knife")]},
        epic_dur={"E": 100.0},
        # Background score 0.1 everywhere: the score is not the GT mask.
        fpv_gt={"F": FpvGT(times=times, score=0.1 + 0.9 * hl, highlight=hl, label="ski")},
        fpv_dur={"F": 20.0},
    )


def test_compute_scores_hand_computed():
    preds = {
        # Captures cut/onion only → coverage 1/2; one exact action → that class AP 1, other 0.
        "E": Prediction([(9.0, 13.05)], [((10.0, 11.0), "cut", "onion", 0.9)], "cooking"),
        # (10, 14.05): frames 10..14, TP 3 → P 3/5, R 1 → F1 0.75; label right.
        "F": Prediction([(10.0, 14.05)], [], "ski"),
    }

    s = compute_scores(
        make_data(), preds, budget_frac=0.145, lam=1.0, theta=0.5, oracle_cov={"E": 1.0}
    )

    assert s["coverage_any"] == pytest.approx(0.5, abs=1e-12)
    assert s["pct_oracle"] == pytest.approx(0.5, abs=1e-12)  # 0.5 / 1.0
    assert s["map"] == pytest.approx(0.5, abs=1e-12)  # (1 + 0) / 2 at every threshold
    assert s["f1"] == pytest.approx(0.75, abs=1e-12)
    assert s["label_acc"] == pytest.approx(1.0, abs=1e-12)


def test_compute_scores_weighted_total_uses_kit_weights():
    preds = {
        # Captures both pairs → coverage 1.0; actions for one of two classes → mAP 0.5.
        "E": Prediction([(9.0, 13.05), (48.0, 52.05)], [((10.0, 11.0), "cut", "onion", 0.9)]),
        "F": Prediction([(10.0, 14.05)], [], "ski"),  # F1 0.75, label 1.0
    }

    s = compute_scores(
        make_data(), preds, budget_frac=0.145, lam=1.0, theta=0.5, oracle_cov={"E": 1.0}
    )

    # 0.40 · 1.0 + 0.25 · 0.5 + 0.25 · 0.75 + 0.10 · 1.0 = 0.8125
    assert s["weighted"] == pytest.approx(0.8125, abs=1e-12)
    assert sum(WEIGHTS.values()) == pytest.approx(1.0, abs=1e-12)


def test_compute_scores_missing_predictions_score_zero():
    s = compute_scores(
        make_data(), {}, budget_frac=0.145, lam=1.0, theta=0.5, oracle_cov={"E": 1.0}
    )

    assert all(v == pytest.approx(0.0, abs=1e-12) for v in s.values())
