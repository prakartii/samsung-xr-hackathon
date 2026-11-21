"""Campaign #3 pair-level scorer: reality.pair_features / pair_scores, epic.pair_weighted,
and the select.pair_* config validation."""

import dataclasses
import math
from types import MappingProxyType

import pytest

from rewind.config import ConfigError, config_from_dict, load_config
from rewind.select.epic import pair_weighted
from rewind.select.reality import (
    FEATURES,
    PAIR_FEATURES,
    cluster_features,
    pair_features,
    pair_scores,
)
from rewind.types import Cluster
from tests.conftest import DEFAULT_CONFIG

NF = len(FEATURES)  # 9 cluster features; pair_prob is column NF, coef index NF + 1
NP = len(PAIR_FEATURES)  # 12


def cl(dist, span=(10.0, 14.0), conf=0.4, members=(0, 1)):
    return Cluster(0, span, members, MappingProxyType(dist), conf)


# un-normalised: 0.6 + 0.2 = 0.8; listed lowest-first to check the descending sort
DIST = {("b", "y"): 0.2, ("a", "x"): 0.6}


def coef_on_pair_prob(b, w):
    """coef = [b, 0 x 9, w, 0, 0]: only pair_prob contributes."""
    coef = [b] + [0.0] * NP
    coef[NF + 1] = w
    return coef


def test_pair_features_hand_computed_rows_and_keys():
    c = cl(DIST)
    x, keys = pair_features([c], 100.0)
    assert x.shape == (2, NP)
    # rows ordered by descending prob: ("a","x") 0.6 first, ("b","y") 0.2 second
    assert keys == [(0, ("a", "x")), (0, ("b", "y"))]
    # pair_prob: 0.6 / 0.8 = 0.75; 0.2 / 0.8 = 0.25
    assert x[:, NF] == pytest.approx([0.75, 0.25])
    # log_pair_rank: log1p(0) = 0; log1p(1) = log 2 = 0.693147
    assert x[:, NF + 1] == pytest.approx([0.0, math.log(2)])
    # pair_ratio = q / top: 0.75 / 0.75 = 1.0; 0.25 / 0.75 = 0.333333
    assert x[:, NF + 2] == pytest.approx([1.0, 1 / 3])
    # first 9 columns = the cluster's own features, repeated on each of its pair rows
    cx = cluster_features([c], 100.0)[0]
    assert x[0, :NF] == pytest.approx(cx)
    assert x[1, :NF] == pytest.approx(cx)


def test_pair_scores_logistic_on_pair_prob_only():
    # coef b = -1, w = 2 on pair_prob; mean 0, std 1 so z = x:
    # ("a","x"): sigmoid(-1 + 2 * 0.75) = sigmoid(0.5) = 0.622459
    # ("b","y"): sigmoid(-1 + 2 * 0.25) = sigmoid(-0.5) = 0.377541
    out = pair_scores([cl(DIST)], 100.0, coef_on_pair_prob(-1.0, 2.0), [0.0] * NP, [1.0] * NP)
    assert len(out) == 1
    assert set(out[0]) == {("a", "x"), ("b", "y")}
    assert out[0][("a", "x")] == pytest.approx(0.622459, abs=1e-6)
    assert out[0][("b", "y")] == pytest.approx(0.377541, abs=1e-6)


def test_pair_scores_one_dict_per_cluster_in_input_order():
    # second cluster has a single pair: q = 1.0 -> sigmoid(-1 + 2 * 1.0) = sigmoid(1) = 0.731059
    cs = [cl(DIST), cl({("c", "z"): 1.0}, span=(50.0, 52.0))]
    out = pair_scores(cs, 100.0, coef_on_pair_prob(-1.0, 2.0), [0.0] * NP, [1.0] * NP)
    assert len(out) == 2
    assert out[1] == pytest.approx({("c", "z"): 0.731059}, abs=1e-6)


def test_pair_scores_and_features_empty_input():
    assert pair_scores([], 100.0, [0.0] * (NP + 1), [0.0] * NP, [1.0] * NP) == []
    x, keys = pair_features([], 100.0)
    assert x.shape == (0, NP)
    assert keys == []


def _cfg(power):
    cfg = load_config(DEFAULT_CONFIG)
    sel = dataclasses.replace(
        cfg.select,
        pair_coef=tuple(coef_on_pair_prob(0.0, 1.0)),
        pair_mean=(0.0,) * NP,
        pair_std=(1.0,) * NP,
        pair_power=power,
    )
    return dataclasses.replace(cfg, select=sel)


def test_pair_weighted_power_one_is_the_learned_probability():
    # b = 0, w = 1: P(a,x) = sigmoid(0.75) = 0.679179; P(b,y) = sigmoid(0.25) = 0.562177
    (out,) = pair_weighted([cl(DIST)], 100.0, _cfg(1.0))
    assert dict(out.pair_dist) == pytest.approx(
        {("a", "x"): 0.679179, ("b", "y"): 0.562177}, abs=1e-6
    )
    assert isinstance(out.pair_dist, MappingProxyType)


def test_pair_weighted_power_two_squares_the_probability():
    # 0.679179 ** 2 = 0.461284; 0.562177 ** 2 = 0.316042
    (out,) = pair_weighted([cl(DIST)], 100.0, _cfg(2.0))
    assert dict(out.pair_dist) == pytest.approx(
        {("a", "x"): 0.461284, ("b", "y"): 0.316042}, abs=1e-6
    )
    # everything else about the cluster is unchanged
    assert (out.span, out.member_idx, out.confidence) == ((10.0, 14.0), (0, 1), 0.4)


def test_config_pair_coef_wrong_length_rejected(default_cfg_dict):
    # NP + 1 = 13 coefficients expected; give 12
    default_cfg_dict["select"]["pair_coef"] = [0.0] * NP
    default_cfg_dict["select"]["pair_mean"] = [0.0] * NP
    default_cfg_dict["select"]["pair_std"] = [1.0] * NP
    with pytest.raises(ConfigError, match="pair"):
        config_from_dict(default_cfg_dict)


def test_config_pair_mean_wrong_length_rejected(default_cfg_dict):
    default_cfg_dict["select"]["pair_coef"] = [0.0] * (NP + 1)
    default_cfg_dict["select"]["pair_mean"] = [0.0] * (NP - 1)
    default_cfg_dict["select"]["pair_std"] = [1.0] * NP
    with pytest.raises(ConfigError, match="pair"):
        config_from_dict(default_cfg_dict)


def test_config_pair_correct_lengths_load(default_cfg_dict):
    default_cfg_dict["select"]["pair_coef"] = [0.0] * (NP + 1)
    default_cfg_dict["select"]["pair_mean"] = [0.0] * NP
    default_cfg_dict["select"]["pair_std"] = [1.0] * NP
    assert len(config_from_dict(default_cfg_dict).select.pair_coef) == NP + 1


def test_config_empty_pair_lists_load_and_mean_off(default_cfg_dict):
    for k in ("pair_coef", "pair_mean", "pair_std"):
        default_cfg_dict["select"][k] = []
    s = config_from_dict(default_cfg_dict).select
    # empty tuple is falsy -> select_epic skips pair_weighted
    assert not s.pair_coef and not s.pair_mean and not s.pair_std
