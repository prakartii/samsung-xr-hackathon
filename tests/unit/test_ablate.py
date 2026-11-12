"""tools/ablate.py: combining tune's EPIC row with its FPVSum fold rows."""

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "ablate.py"
_spec = importlib.util.spec_from_file_location("ablate", _PATH)
tool = importlib.util.module_from_spec(_spec)
sys.modules["ablate"] = tool
_spec.loader.exec_module(tool)


def test_combine_tune_averages_folds_and_reweights():
    # EPIC: coverage 0.5, mAP 0.4. Folds: f1 0.2 and 0.4 -> 0.3; label 1.0 and 0.5 -> 0.75.
    # weighted = 0.40 * 0.5 + 0.25 * 0.4 + 0.25 * 0.3 + 0.10 * 0.75
    #          = 0.2 + 0.1 + 0.075 + 0.075 = 0.45
    epic = {"coverage_any": "0.5", "coverage_frac": "0.3", "pct_oracle": "0.6", "map": "0.4"}
    folds = [{"f1": "0.2", "label_acc": "1.0"}, {"f1": "0.4", "label_acc": "0.5"}]
    out = tool.combine_tune(epic, folds)
    assert out["f1"] == pytest.approx(0.3, abs=1e-12)
    assert out["label_acc"] == pytest.approx(0.75, abs=1e-12)
    assert out["weighted"] == pytest.approx(0.45, abs=1e-12)
    assert out["coverage_any"] == pytest.approx(0.5, abs=1e-12)
