"""Metric floors on `tune` (CLAUDE.md §7.3.12): regenerate, score, compare.

Runs tools/predict_cached.py on the whole `tune` split (~3 min on a laptop CPU) and
scores it with rewind_eval.score. Needs the feature cache (cache/, Kaggle logs),
weights/ and data/annotations.
"""

import json
import subprocess
import sys

import pytest
import yaml

from rewind_eval.score import main as score_main
from tests.conftest import REPO_ROOT

FLOORS = yaml.safe_load(
    (REPO_ROOT / "tests" / "integration" / "floors.yaml").read_text(encoding="utf-8")
)
N_FOLDS = 5

pytestmark = [
    pytest.mark.slow,
    pytest.mark.data,
    pytest.mark.skipif(
        not list((REPO_ROOT / "cache" / "kaggle_logs").glob("shard_*_tune.jsonl"))
        or not (REPO_ROOT / "data" / "annotations" / "fpvsum" / "gt_data.mat").exists()
        or not (REPO_ROOT / "weights" / "actionformer_epic_verb.pth.tar").exists(),
        reason="needs the tune feature cache, annotations and weights",
    ),
]


def score(capsys, pred, *extra) -> dict:
    capsys.readouterr()
    score_main(["--split", "tune", "--pred", str(pred), "--tag", "floors", "--no-write", *extra])
    return json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def tune_pred(tmp_path_factory):
    out = tmp_path_factory.mktemp("full_tune")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "predict_cached.py"),
         "--split", "tune", "--out", str(out)],
        check=True, capture_output=True,
    )  # fmt: skip
    return out


def test_tune_epic_metrics_stay_above_floors(tune_pred, capsys):
    row = score(capsys, tune_pred)
    tol = FLOORS["tolerance"]
    low = {k: (float(row[k]), f) for k, f in FLOORS["tune_epic"].items() if float(row[k]) < f - tol}
    assert low == {}, f"below floor (got, floor): {low}"


def test_tune_fpvsum_cv_metrics_stay_above_floors(tune_pred, capsys):
    rows = [score(capsys, tune_pred, "--fold", str(k)) for k in range(N_FOLDS)]
    tol = FLOORS["tolerance"]
    low = {}
    for k, f in FLOORS["tune_fpvsum_cv"].items():
        mean = sum(float(r[k]) for r in rows) / N_FOLDS
        if mean < f - tol:
            low[k] = (mean, f)
    assert low == {}, f"below floor (got, floor): {low}"
