"""Scoring on the real annotations (marker `data`; run tools/fetch_annotations.sh)."""

import pytest

from rewind_eval.score import baseline_predictions, compute_scores, load_eval_data
from rewind_eval.splits import load_splits
from tests.conftest import REPO_ROOT

ANN = REPO_ROOT / "data" / "annotations"
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (ANN / "epic" / "EPIC_100_validation.csv").exists()
        or not (ANN / "fpvsum" / "gt_data.mat").exists(),
        reason="run tools/fetch_annotations.sh",
    ),
]


def test_uniform_baseline_on_report_reproduces_the_leaderboard_row():
    # Leaderboard (P2): uniform on report → coverage 0.3380 (any), 0.1768 (fraction).
    # F1 was 0.1626 over 49 videos; 0.1645 once 3 mistimed GTs left F1 (2026-11-12);
    # 0.1669 over the 44 usable of the 47 the organizers score (Q14, 2026-11-20).
    data = load_eval_data("report")
    preds = baseline_predictions(data, "uniform", 0.145)

    s = compute_scores(data, preds, budget_frac=0.145, lam=1.0, theta=0.5, oracle_cov={})

    assert len(data.fpv_gt) == 47
    assert s["coverage_any"] == pytest.approx(0.3380, abs=5e-5)
    assert s["coverage_frac"] == pytest.approx(0.1768, abs=5e-5)
    assert s["f1"] == pytest.approx(0.1669, abs=5e-5)


def test_tune_uses_only_the_requested_fpvsum_fold():
    folds = load_splits().fpvsum_folds

    assert load_eval_data("tune").fpv_gt == {}
    assert set(load_eval_data("tune", fold=2).fpv_gt) == {v for v, k in folds.items() if k == 2}
