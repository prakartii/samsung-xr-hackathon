"""select/learned_importance.py: per-step features and the learned importance logit."""

from dataclasses import replace

import numpy as np
import pytest

from rewind.analyze import fpvsum_highlights
from rewind.config import ConfigError, config_from_dict, load_config
from rewind.select.learned_importance import (
    feature_names,
    learned_importance,
    local_max,
    step_features,
)
from tests.conftest import DEFAULT_CONFIG


def test_feature_names_counts_motion_position_and_prompts():
    # 5 base signals x 8 context columns + 3 position columns + 1 per prompt
    assert len(feature_names(0)) == 43
    assert len(feature_names(4)) == 47
    assert feature_names(2)[-2:] == ("txt0", "txt1")


def test_local_max_takes_the_max_over_the_window_with_edge_padding():
    # half 1: windows [1,1,3] [1,3,2] [3,2,0] [2,0,0] [0,0,0] (edges repeated)
    assert local_max(np.array([1.0, 3.0, 2.0, 0.0, 0.0]), 1) == pytest.approx([3, 3, 3, 2, 0])


def grid(n=8, rate=4.0):
    t = np.arange(n) / rate
    tc = np.arange(int(np.ceil(n / rate))).astype(float)
    return t, tc


def test_step_features_raw_columns_are_zscored_flow_and_raw_hands():
    t = np.array([0.0, 1.0, 2.0])  # clip frames on the same grid
    emb = np.tile(np.eye(4)[0], (3, 1))
    x = step_features(t, [1.0, 2.0, 3.0], [0.0, 0.5, 1.0], [0.0, 0.0, 0.0], t, emb,
                      np.zeros((3, 0)), 3.0)  # fmt: skip
    names = feature_names(0)
    # z(1, 2, 3): mean 2, population std sqrt(2/3) -> -1.2247, 0, 1.2247
    assert x[:, names.index("flow")] == pytest.approx([-1.224745, 0.0, 1.224745], abs=1e-5)
    assert x[:, names.index("hands")] == pytest.approx([0.0, 0.5, 1.0])
    assert x[:, names.index("gm")] == pytest.approx([0.0, 0.0, 0.0])  # constant -> 0


def test_step_features_position_columns():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    emb = np.tile(np.eye(4)[0], (4, 1))
    x = step_features(t, np.zeros(4), np.zeros(4), np.zeros(4), t, emb, np.zeros((4, 0)), 4.0)
    names = feature_names(0)
    # pos = t / 4; edge = min(pos, 1 - pos); pos2 = pos^2
    assert x[:, names.index("pos")] == pytest.approx([0.0, 0.25, 0.5, 0.75])
    assert x[:, names.index("edge")] == pytest.approx([0.0, 0.25, 0.5, 0.25])
    assert x[:, names.index("pos2")] == pytest.approx([0.0, 0.0625, 0.25, 0.5625])


def test_step_features_prompt_column_follows_the_prompt_similarity():
    t, tc = grid(40, 4.0)  # 10 s at 4 fps, clip frames at 0..9 s
    emb = np.tile(np.eye(4)[0], (len(tc), 1))
    sims = np.stack([np.zeros(len(tc)), np.arange(len(tc), dtype=float)], axis=1)
    x = step_features(t, np.zeros(40), np.zeros(40), np.zeros(40), tc, emb, sims, 10.0)
    names = feature_names(2)
    assert x[:, names.index("txt0")] == pytest.approx(np.zeros(40))  # constant -> z = 0
    rising = x[:, names.index("txt1")]
    assert rising[-1] > 0 > rising[0]  # z of a rising similarity: low early, high late


def test_learned_importance_is_the_standardised_logit():
    # z = ([[1,2],[3,4]] - [1,2]) / [2, 1 (std 0 -> 1)] = [[0,0],[1,2]]
    # logit = 0.5 + 1*z0 - 1*z1 -> [0.5, 0.5 + 1 - 2 = -0.5]; sigma 0 = no smoothing
    out = learned_importance(np.array([[1.0, 2.0], [3.0, 4.0]]), [0.5, 1.0, -1.0],
                             [1.0, 2.0], [2.0, 0.0], 0.0)  # fmt: skip
    assert out == pytest.approx([0.5, -0.5])


def test_config_learned_importance_coefficients_must_match_the_feature_count(default_cfg_dict):
    default_cfg_dict["learned_importance"]["coef"] = [0.1, 0.2]
    with pytest.raises(ConfigError, match="learned_importance"):
        config_from_dict(default_cfg_dict)


def test_config_empty_learned_importance_coefficients_mean_off(default_cfg_dict):
    for k in ("coef", "mean", "std"):
        default_cfg_dict["learned_importance"][k] = []
    assert config_from_dict(default_cfg_dict).learned_importance.coef == ()


def flat_signals(duration=100.0):
    t = np.arange(int(duration * 4)) / 4.0
    tc = np.arange(int(duration), dtype=float)
    z = np.zeros(len(t))
    emb = np.tile(np.eye(8)[0], (len(tc), 1))
    return {"t": t, "flow": z, "global_motion": z}, {"t": t, "hands": z}, {"t": tc, "clip_emb": emb}


def cfg_weighting(name, w):
    cfg = load_config(DEFAULT_CONFIG)
    li = cfg.learned_importance
    names = feature_names(len(li.prompts))
    coef = [0.0] * (len(names) + 1)
    coef[1 + names.index(name)] = w
    li = replace(li, coef=tuple(coef), mean=(0.0,) * len(names), std=(1.0,) * len(names))
    return replace(cfg, learned_importance=li)


@pytest.mark.parametrize(("w", "late"), [(1.0, True), (-1.0, False)])
def test_fpvsum_highlights_learned_importance_follows_its_weights(w, late):
    # flat motion and CLIP: only the position weight differs, so +pos puts every
    # highlight in the second half of the video and -pos in the first half
    flow, hands, clip = flat_signals()
    cfg = cfg_weighting("pos", w)
    prompts = np.tile(np.eye(8)[1], (len(cfg.learned_importance.prompts), 1))
    segs, _, _ = fpvsum_highlights(flow, hands, clip, 100.0, cfg, None, prompts)
    assert segs
    assert all((s >= 50.0) if late else (e <= 50.0) for s, e in segs)


def test_fpvsum_highlights_without_prompt_embeddings_uses_hand_weights():
    flow, hands, clip = flat_signals()
    cfg = cfg_weighting("pos", 1.0)
    _, _, (_, imp_learned) = fpvsum_highlights(flow, hands, clip, 100.0, cfg, None, None)
    off = load_config(DEFAULT_CONFIG)
    _, _, (_, imp_hand) = fpvsum_highlights(flow, hands, clip, 100.0, off, None, None)
    assert imp_learned == pytest.approx(imp_hand)
