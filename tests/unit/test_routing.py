"""Unit tests for signal-only routing (PROGRESS.md P3: clear EPIC, clear FPVSum, uncertain)."""

import numpy as np
import pytest

from rewind.routing import route

THR = 0.5
MIN_CONF = 0.1


def test_route_clear_cooking_goes_to_epic():
    # mean(0.8, 0.9, 1.0) = 0.9; margin = |0.9 - 0.5| = 0.4 >= 0.1
    d = route(np.array([0.8, 0.9, 1.0]), THR, MIN_CONF)
    assert d.dataset == "epic_kitchens_100"
    assert d.route_conf == pytest.approx(0.4, abs=1e-9)
    assert d.p_cooking == pytest.approx(0.9, abs=1e-9)


def test_route_clear_non_cooking_goes_to_fpvsum():
    # mean(0.0, 0.1, 0.2) = 0.1; margin = 0.4
    d = route(np.array([0.0, 0.1, 0.2]), THR, MIN_CONF)
    assert d.dataset == "fpvsum"
    assert d.route_conf == pytest.approx(0.4, abs=1e-9)


def test_route_small_margin_is_uncertain():
    # mean(0.5, 0.6) = 0.55; margin = 0.05 < 0.1 -> uncertain
    d = route(np.array([0.5, 0.6]), THR, MIN_CONF)
    assert d.dataset == "unknown"
    assert d.route_conf == pytest.approx(0.05, abs=1e-9)
    # The leaning is still recoverable for the uncertain-route label rule.
    assert d.p_cooking >= THR


def test_route_margin_exactly_min_conf_is_confident():
    # mean = 0.75 -> margin = 0.25; with min_conf = 0.25 the route is not uncertain
    # (uncertain is strictly below min_conf).
    d = route(np.array([0.75]), THR, 0.25)
    assert d.dataset == "epic_kitchens_100"


def test_route_mean_exactly_at_threshold_with_zero_min_conf_goes_to_epic():
    # P(cooking) >= thr -> EPIC (ties go to EPIC so a cooking video keeps its actions)
    d = route(np.array([0.4, 0.6]), THR, 0.0)
    assert d.dataset == "epic_kitchens_100"


def test_route_uses_mean_not_majority():
    # 2 of 3 frames above 0.5, but mean(0.55, 0.55, 0.0) = 0.3667 -> fpvsum
    d = route(np.array([0.55, 0.55, 0.0]), THR, MIN_CONF)
    assert d.dataset == "fpvsum"


@pytest.mark.parametrize("bad", [np.array([]), np.array([0.5, 1.2]), np.array([np.nan])])
def test_route_invalid_probabilities_raise(bad):
    with pytest.raises(ValueError):
        route(bad, THR, MIN_CONF)
