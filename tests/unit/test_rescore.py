"""Campaign #4 output-action rescorer: rescore.action_features / rescore_probs /
rescore_actions, output_actions with and without it, and the actions.rescore_* config."""

import dataclasses
import math

import numpy as np
import pytest

from rewind.analyze import output_actions
from rewind.config import ConfigError, config_from_dict, load_config
from rewind.detect.pairing import VNPrior
from rewind.detect.rescore import FEATURES, action_features, rescore_actions, rescore_probs
from rewind.types import Proposal
from tests.conftest import DEFAULT_CONFIG

NA = len(FEATURES)  # 16

ONE = Proposal(
    (2.0, 6.0),
    (("take", 0.9), ("put", 0.4)),
    (("knife", 0.8),),
    # listed lowest-first to check the score sort
    (("put", "knife", 0.3), ("take", "knife", 0.5)),
)
NOUN_DETS = [((3.0, 6.0), "knife", 0.8)]


def test_action_features_single_proposal_hand_computed_rows():
    # take knife (0.5) first, then put knife (0.3).
    # vn_tiou: (2, 6) vs (3, 6) = 3 / 4 = 0.75; log duration log 4; position = mid 4 / last
    # end 6 = 2/3; no other proposal overlaps; knife appears twice -> log 3, each verb and
    # each class once -> log 2; score ratio 0.5 / 0.5 = 1 and 0.3 / 0.5 = 0.6.
    # put is verb rank 1 and the pair was listed at index 0 -> log 2 / log 1 = 0.
    acts, x = action_features([ONE], NOUN_DETS, 0.0)

    assert [(v, n, s) for _, v, n, s in acts] == [("take", "knife", 0.5), ("put", "knife", 0.3)]
    l2, l3, l4 = math.log(2), math.log(3), math.log(4)
    assert x[0] == pytest.approx(
        [0.5, 0.9, 0.8, 0, 0, math.log(2), 0.75, l4, 2 / 3, 0, 0, l2, l2, l3, 1.0, 0], abs=1e-12
    )
    assert x[1] == pytest.approx(
        [0.3, 0.4, 0.8, l2, 0, 0, 0.75, l4, 2 / 3, 0, 0, l2, l2, l3, 0.6, 0], abs=1e-12
    )


def test_action_features_second_same_class_action_gets_rank_and_duplicate_tiou():
    # take knife on (0, 4) at 0.5 and on (2, 6) at 0.3: the later one is class rank 1
    # (log 2), class count 2 (log 3), and overlaps the first by tIoU 2 / 6 = 1/3.
    a = Proposal((0.0, 4.0), (("take", 0.9),), (("knife", 0.8),), (("take", "knife", 0.5),))
    b = Proposal((2.0, 6.0), (("take", 0.6),), (("knife", 0.5),), (("take", "knife", 0.3),))
    _, x = action_features([a, b], [], 0.0)
    col = {f: i for i, f in enumerate(FEATURES)}

    assert x[:, col["log_class_rank"]] == pytest.approx([0.0, math.log(2)], abs=1e-12)
    assert x[:, col["dup_tiou"]] == pytest.approx([0.0, 1 / 3], abs=1e-12)
    assert x[:, col["log_overlap"]] == pytest.approx([math.log(2)] * 2, abs=1e-12)
    assert x[:, col["vn_tiou"]] == pytest.approx([0.0, 0.0], abs=1e-12)  # no noun dets


def test_action_features_min_score_drops_low_pairs():
    acts, x = action_features([ONE], NOUN_DETS, 0.4)
    assert [(v, n) for _, v, n, _ in acts] == [("take", "knife")]
    assert x.shape == (1, NA)


def test_rescore_probs_is_sigmoid_of_standardized_linear_score():
    # b = 0.5, w_score = 2, mean 0.3, std 0.1: (0.5 - 0.3) / 0.1 = 2 -> 0.5 + 4 = 4.5
    coef = [0.5, 2.0] + [0.0] * (NA - 1)
    x = np.zeros((1, NA))
    x[0, 0] = 0.5
    p = rescore_probs(x, coef, [0.3] + [0.0] * (NA - 1), [0.1] + [1.0] * (NA - 1))
    assert p == pytest.approx([1 / (1 + math.exp(-4.5))], abs=1e-12)


def test_rescore_actions_negative_weight_reverses_order_and_scales_by_old_power():
    # w_score = -10, mean 0, std 1: P(take) = sigmoid(-5) = 0.0066929,
    # P(put) = sigmoid(-3) = 0.0474259 -> put first. old_power 1 multiplies by the pairing
    # score: put 0.0474259 * 0.3 = 0.0142278 > take 0.0066929 * 0.5 = 0.0033464.
    coef = [0.0, -10.0] + [0.0] * (NA - 1)
    out = rescore_actions([ONE], NOUN_DETS, 0.0, coef, [0.0] * NA, [1.0] * NA, 1.0, 2000)

    assert [(v, s) for _, v, _, s in out] == [
        ("put", pytest.approx(0.0474259 * 0.3, abs=1e-7)),
        ("take", pytest.approx(0.0066929 * 0.5, abs=1e-7)),
    ]


def test_rescore_actions_caps_at_max_per_video():
    # w_score = 1: P(take) = sigmoid(0.5) = 0.6224593 > P(put) = sigmoid(0.3); keep 1
    coef = [0.0, 1.0] + [0.0] * (NA - 1)
    out = rescore_actions([ONE], NOUN_DETS, 0.0, coef, [0.0] * NA, [1.0] * NA, 0.0, 1)
    assert [(v, s) for _, v, _, s in out] == [("take", pytest.approx(0.6224593, abs=1e-7))]


PRIOR = VNPrior.from_counts(("put", "take"), ("knife",), np.ones((2, 1)), eps=1.0)
DETS = ([((2.0, 6.0), "take", 0.9), ((2.0, 6.0), "put", 0.4)], [((3.0, 6.0), "knife", 0.8)])


def _cfg(coef):
    cfg = load_config(DEFAULT_CONFIG)
    n = len(coef) - 1 if coef else 0
    act = dataclasses.replace(
        cfg.actions,
        rescore_coef=tuple(coef),
        rescore_mean=(0.0,) * n,
        rescore_std=(1.0,) * n,
        rescore_old_power=0.0,
    )
    return dataclasses.replace(cfg, actions=act)


def test_output_actions_empty_rescore_keeps_pairing_scores():
    # prior_alpha 0, power 0.25: take (0.9 * 0.8) ** 0.25, put (0.4 * 0.8) ** 0.25
    out = output_actions(DETS, PRIOR, _cfg(()))
    assert [(v, s) for _, v, _, s in out] == [
        ("take", pytest.approx(0.72**0.25, abs=1e-12)),
        ("put", pytest.approx(0.32**0.25, abs=1e-12)),
    ]


def test_output_actions_with_rescore_uses_learned_probability():
    # w on verb_prob = -10 (mean 0, std 1): P(take) = sigmoid(-9) = 1.2339e-4,
    # P(put) = sigmoid(-4) = 0.0179862 -> put first, scored P (old_power 0)
    coef = [0.0, 0.0, -10.0] + [0.0] * (NA - 2)
    out = output_actions(DETS, PRIOR, _cfg(coef))
    assert [(v, s) for _, v, _, s in out] == [
        ("put", pytest.approx(0.0179862, abs=1e-7)),
        ("take", pytest.approx(1.2339e-4, abs=1e-8)),
    ]


def test_config_rescore_coef_wrong_length_rejected(default_cfg_dict):
    # NA + 1 = 17 coefficients expected; give 16
    default_cfg_dict["actions"]["rescore_coef"] = [0.0] * NA
    default_cfg_dict["actions"]["rescore_mean"] = [0.0] * NA
    default_cfg_dict["actions"]["rescore_std"] = [1.0] * NA
    with pytest.raises(ConfigError, match="rescore"):
        config_from_dict(default_cfg_dict)


def test_config_rescore_std_wrong_length_rejected(default_cfg_dict):
    default_cfg_dict["actions"]["rescore_coef"] = [0.0] * (NA + 1)
    default_cfg_dict["actions"]["rescore_mean"] = [0.0] * NA
    default_cfg_dict["actions"]["rescore_std"] = [1.0] * (NA - 1)
    with pytest.raises(ConfigError, match="rescore"):
        config_from_dict(default_cfg_dict)


def test_config_rescore_correct_lengths_load(default_cfg_dict):
    default_cfg_dict["actions"]["rescore_coef"] = [0.0] * (NA + 1)
    default_cfg_dict["actions"]["rescore_mean"] = [0.0] * NA
    default_cfg_dict["actions"]["rescore_std"] = [1.0] * NA
    assert len(config_from_dict(default_cfg_dict).actions.rescore_coef) == NA + 1


def test_config_negative_rescore_old_power_rejected(default_cfg_dict):
    default_cfg_dict["actions"]["rescore_old_power"] = -0.5
    with pytest.raises(ConfigError, match="rescore_old_power"):
        config_from_dict(default_cfg_dict)
