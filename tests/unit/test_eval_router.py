"""tools/eval_router.py: router agreement with the manifest's dataset column."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "eval_router.py"
_spec = importlib.util.spec_from_file_location("eval_router", _PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_agreement_counts_correct_wrong_and_uncertain():
    # 4 videos: 2 routed right, 1 routed to the wrong dataset, 1 uncertain ("unknown").
    routed = {"a": "epic_kitchens_100", "b": "fpvsum", "c": "fpvsum", "d": "unknown"}
    truth = {"a": "epic_kitchens_100", "b": "fpvsum", "c": "epic_kitchens_100", "d": "fpvsum"}

    stats = tool.agreement(routed, truth)

    # agreement = correct / all = 2 / 4; uncertain videos count against it
    assert stats["agreement"] == pytest.approx(0.5, abs=1e-12)
    assert stats["wrong"] == ["c"]
    assert stats["uncertain"] == ["d"]


def test_agreement_perfect_routing_is_one():
    routed = {"a": "fpvsum", "b": "epic_kitchens_100"}
    assert tool.agreement(routed, dict(routed))["agreement"] == pytest.approx(1.0, abs=1e-12)
