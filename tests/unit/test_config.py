"""configs/default.yaml → typed Config: strict keys, strict types, range checks, hash."""

import pytest
import yaml

from rewind.config import ConfigError, config_from_dict, config_hash, load_config
from tests.conftest import DEFAULT_CONFIG


def test_load_config_default_file_loads_typed_values():
    cfg = load_config(DEFAULT_CONFIG)

    assert cfg.select.window_lengths == pytest.approx((4.05, 5.0, 6.0), abs=1e-9)
    assert isinstance(cfg.select.window_lengths, tuple)
    assert cfg.post.short_video_mode == "min_seg"
    assert cfg.actions.max_per_video == 2000


def test_config_int_written_for_float_field_becomes_float(default_cfg_dict):
    # YAML `stride: 1` parses as int 1; a float field must accept it as 1.0.
    default_cfg_dict["select"]["stride"] = 1

    cfg = config_from_dict(default_cfg_dict)

    assert isinstance(cfg.select.stride, float)
    assert cfg.select.stride == pytest.approx(1.0, abs=1e-12)


def test_config_unknown_key_is_rejected_with_its_path(default_cfg_dict):
    default_cfg_dict["select"]["budget_fraction"] = 0.1  # typo of budget_frac

    with pytest.raises(ConfigError, match=r"select: unknown key\(s\) \['budget_fraction'\]"):
        config_from_dict(default_cfg_dict)


def test_config_unknown_section_is_rejected(default_cfg_dict):
    default_cfg_dict["selct"] = {}

    with pytest.raises(ConfigError, match=r"<root>: unknown key\(s\) \['selct'\]"):
        config_from_dict(default_cfg_dict)


def test_config_missing_key_is_rejected_with_its_path(default_cfg_dict):
    del default_cfg_dict["hedge"]["share"]

    with pytest.raises(ConfigError, match=r"hedge: missing key\(s\) \['share'\]"):
        config_from_dict(default_cfg_dict)


def test_config_string_for_number_is_rejected(default_cfg_dict):
    default_cfg_dict["select"]["stride"] = "0.5"

    with pytest.raises(ConfigError, match=r"select\.stride: expected a number"):
        config_from_dict(default_cfg_dict)


def test_config_bool_for_integer_is_rejected(default_cfg_dict):
    # bool is a subclass of int in Python; `true` must not pass as 1.
    default_cfg_dict["run"]["seed"] = True

    with pytest.raises(ConfigError, match=r"run\.seed: expected an integer"):
        config_from_dict(default_cfg_dict)


def test_config_float_for_integer_is_rejected(default_cfg_dict):
    default_cfg_dict["cluster"]["k"] = 5.5

    with pytest.raises(ConfigError, match=r"cluster\.k: expected an integer"):
        config_from_dict(default_cfg_dict)


def test_config_bad_list_item_is_rejected_with_its_index(default_cfg_dict):
    default_cfg_dict["select"]["window_lengths"] = [5.0, "six"]

    with pytest.raises(ConfigError, match=r"select\.window_lengths\[1\]: expected a number"):
        config_from_dict(default_cfg_dict)


def test_config_unknown_capture_rule_is_rejected(default_cfg_dict):
    default_cfg_dict["select"]["capture_rule"] = "tiou"

    with pytest.raises(ConfigError, match=r"select\.capture_rule: 'tiou' is not one of"):
        config_from_dict(default_cfg_dict)


def test_config_unknown_short_video_mode_is_rejected(default_cfg_dict):
    default_cfg_dict["post"]["short_video_mode"] = "skip"

    with pytest.raises(ConfigError, match=r"post\.short_video_mode: 'skip' is not one of"):
        config_from_dict(default_cfg_dict)


def test_config_empty_team_id_is_rejected(default_cfg_dict):
    default_cfg_dict["team"]["team_id"] = ""

    with pytest.raises(ConfigError, match=r"team\.team_id: expected a non-empty string"):
        config_from_dict(default_cfg_dict)


@pytest.mark.parametrize("budget", [0.151, 0.0, -0.1])
def test_config_budget_outside_kit_limit_is_rejected(default_cfg_dict, budget):
    # Kit budget is 15% of T: budget_frac must be in (0, 0.15].
    default_cfg_dict["select"]["budget_frac"] = budget

    with pytest.raises(ConfigError, match=r"select\.budget_frac"):
        config_from_dict(default_cfg_dict)


def test_config_budget_at_kit_limit_is_accepted(default_cfg_dict):
    default_cfg_dict["select"]["budget_frac"] = 0.15

    assert config_from_dict(default_cfg_dict).select.budget_frac == pytest.approx(0.15, abs=1e-12)


def test_config_window_shorter_than_min_emit_is_rejected(default_cfg_dict):
    # 4.0 < MIN_SEG_EMIT (4.05): rounding could push it under the kit's 4.0 s minimum.
    default_cfg_dict["select"]["window_lengths"] = [4.0, 5.0]

    with pytest.raises(ConfigError, match=r"select\.window_lengths: \[4\.0\] below MIN_SEG_EMIT"):
        config_from_dict(default_cfg_dict)


def test_config_empty_window_lengths_is_rejected(default_cfg_dict):
    default_cfg_dict["select"]["window_lengths"] = []

    with pytest.raises(ConfigError, match=r"select\.window_lengths: must not be empty"):
        config_from_dict(default_cfg_dict)


@pytest.mark.parametrize("n", [0, 2001])
def test_config_max_actions_outside_kit_cap_is_rejected(default_cfg_dict, n):
    default_cfg_dict["actions"]["max_per_video"] = n

    with pytest.raises(ConfigError, match=r"actions\.max_per_video"):
        config_from_dict(default_cfg_dict)


def test_config_threshold_above_one_is_rejected(default_cfg_dict):
    default_cfg_dict["routing"]["cooking_thr"] = 1.5

    with pytest.raises(ConfigError, match=r"routing\.cooking_thr: 1\.5 must be in \[0, 1\]"):
        config_from_dict(default_cfg_dict)


def test_config_importance_window_order_is_enforced(default_cfg_dict):
    default_cfg_dict["importance"]["min_window"] = 12.0  # > max_window (10.0)

    with pytest.raises(ConfigError, match=r"importance\.min_window/max_window"):
        config_from_dict(default_cfg_dict)


def test_config_hash_ignores_comments_and_key_order(default_cfg_dict, tmp_path):
    reordered = dict(reversed(list(default_cfg_dict.items())))
    text = "# a comment that must not change the hash\n" + yaml.safe_dump(reordered)
    (tmp_path / "c.yaml").write_text(text, encoding="utf-8")
    reloaded = yaml.safe_load((tmp_path / "c.yaml").read_text(encoding="utf-8"))

    assert config_hash(reloaded) == config_hash(default_cfg_dict)


def test_config_hash_changes_when_a_value_changes(default_cfg_dict):
    before = config_hash(default_cfg_dict)
    default_cfg_dict["select"]["stride"] = 0.25

    assert config_hash(default_cfg_dict) != before
