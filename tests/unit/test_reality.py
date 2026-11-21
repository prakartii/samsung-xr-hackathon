"""select/reality.py: cluster features and the learned P(real) (Campaign #2)."""

import math
from types import MappingProxyType

import numpy as np
import pytest

from rewind.select.reality import FEATURES, cluster_features, reality_scores
from rewind.types import Cluster


def cl(span, dist, conf, members=(0,)):
    return Cluster(0, span, members, MappingProxyType(dist), conf)


def test_cluster_features_hand_computed_row():
    # one cluster (10, 14) in a 100 s video, 2 members, dist .75/.25, confidence .4:
    # log_members log(3); log_duration log(4); entropy of (.75, .25) / log 2 = 0.81128;
    # top1 .75; margin .5; position 12/100; density: no other cluster -> log1p(0) = 0;
    # conf_rank: single cluster -> 0
    x = cluster_features(
        [cl((10.0, 14.0), {("a", "x"): 0.75, ("b", "y"): 0.25}, 0.4, (0, 1))], 100.0
    )
    assert x.shape == (1, len(FEATURES))
    expected = [0.4, math.log(3), math.log(4), 0.811278, 0.75, 0.5, 0.12, 0.0, 0.0]
    assert x[0] == pytest.approx(expected, abs=1e-5)


def test_cluster_features_density_and_rank():
    # clusters at mids 10, 20, 100: the first two are within 15 s of each other
    cs = [cl((9.0, 11.0), {("a", "x"): 1.0}, 0.2), cl((19.0, 21.0), {("a", "x"): 1.0}, 0.6),
          cl((99.0, 101.0), {("a", "x"): 1.0}, 0.4)]  # fmt: skip
    x = cluster_features(cs, 200.0)
    assert x[:, 7] == pytest.approx([math.log(2), math.log(2), 0.0])
    assert x[:, 8] == pytest.approx([0.0, 1.0, 0.5])  # confidence rank 0.2 < 0.4 < 0.6


def test_reality_scores_is_a_logistic_of_standardised_features():
    c = [cl((10.0, 14.0), {("a", "x"): 1.0}, 0.4)]
    x = cluster_features(c, 100.0)[0]
    coef = [0.5] + [0.0] * len(FEATURES)
    coef[1] = 2.0  # weight on confidence only
    mean, std = list(x), [1.0] * len(FEATURES)
    mean[0] = 0.0  # so the confidence term is 2 * 0.4 = 0.8
    p = reality_scores(c, 100.0, coef, mean, std)
    assert p[0] == pytest.approx(1 / (1 + math.exp(-(0.5 + 0.8))))


def test_reality_scores_empty_input():
    assert reality_scores([], 100.0, [0.0] * 10, [0.0] * 9, [1.0] * 9).shape == (0,)


def test_reality_rows_follow_input_order():
    cs = [cl((50.0, 52.0), {("a", "x"): 1.0}, 0.9), cl((1.0, 3.0), {("a", "x"): 1.0}, 0.1)]
    x = cluster_features(cs, 100.0)
    assert x[0, 0] == pytest.approx(0.9) and x[1, 0] == pytest.approx(0.1)
    assert np.isfinite(x).all()
