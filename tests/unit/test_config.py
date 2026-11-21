"""configs/default.yaml → typed Config: strict keys, strict types, range checks, hash."""

import pytest

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


def test_config_unknown_select_mode_is_rejected(default_cfg_dict):
    default_cfg_dict["select"]["mode"] = "greedy"

    with pytest.raises(ConfigError, match=r"select\.mode: 'greedy' is not one of"):
        config_from_dict(default_cfg_dict)


def test_config_select_mode_defaults_to_expected_coverage(default_cfg_dict):
    assert config_from_dict(default_cfg_dict).select.mode == "expected"


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


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("signals", "dense_fps", float("nan")),
        ("select", "stride", float("inf")),
        ("importance", "max_window", float("-inf")),
    ],
)
def test_config_non_finite_number_is_rejected(default_cfg_dict, section, key, value):
    # NaN slips past every `v <= 0` style check, so it must be stopped at the type level.
    default_cfg_dict[section][key] = value

    with pytest.raises(ConfigError, match=rf"{section}\.{key}: expected a finite number"):
        config_from_dict(default_cfg_dict)


def test_config_nan_window_length_is_rejected(default_cfg_dict):
    # `nan < 4.05` is False, so without the finite check this would pass the 4 s guard.
    default_cfg_dict["select"]["window_lengths"] = [float("nan")]

    with pytest.raises(ConfigError, match=r"select\.window_lengths\[0\]: expected a finite"):
        config_from_dict(default_cfg_dict)


def test_load_config_duplicate_key_is_rejected(tmp_path):
    # A tuning edit that adds a second `stride:` must not silently win.
    text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
        "  stride: 0.5", "  stride: 0.5\n  stride: 99.0", 1
    )
    path = tmp_path / "dup.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match=r"duplicate key 'stride'"):
        load_config(path)


def test_config_non_mapping_section_is_rejected(default_cfg_dict):
    default_cfg_dict["select"] = 5

    with pytest.raises(ConfigError, match=r"select: expected a mapping, got int"):
        config_from_dict(default_cfg_dict)


def test_config_scalar_for_list_is_rejected(default_cfg_dict):
    default_cfg_dict["select"]["window_lengths"] = 5.0

    with pytest.raises(ConfigError, match=r"select\.window_lengths: expected a list"):
        config_from_dict(default_cfg_dict)


def test_config_non_string_team_id_is_rejected(default_cfg_dict):
    default_cfg_dict["team"]["team_id"] = 7

    with pytest.raises(ConfigError, match=r"team\.team_id: expected a non-empty string"):
        config_from_dict(default_cfg_dict)


def test_config_importance_min_window_below_min_emit_is_rejected(default_cfg_dict):
    # 4.0 < MIN_SEG_EMIT (4.05): FPVSum windows must clear the kit's 4 s minimum too.
    default_cfg_dict["importance"]["min_window"] = 4.0

    with pytest.raises(ConfigError, match=r"importance\.min_window/max_window"):
        config_from_dict(default_cfg_dict)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("select", "stride", 0.0, r"select\.stride: 0\.0 must be > 0"),
        ("signals", "dense_fps", -1.0, r"signals\.dense_fps: -1\.0 must be > 0"),
        ("pairing", "eps", 0.0, r"pairing\.eps: 0\.0 must be > 0"),
        ("cluster", "k", 0, r"cluster\.k: 0 must be > 0"),
        ("post", "merge_gap", -5.0, r"post\.merge_gap: -5\.0 must be >= 0"),
        ("post", "merge_gap_epic", -1.0, r"post\.merge_gap_epic: -1\.0 must be >= 0"),
        ("hedge", "min_spacing", -1.0, r"hedge\.min_spacing: -1\.0 must be >= 0"),
        ("pairing", "alpha", -0.5, r"pairing\.alpha: -0\.5 must be >= 0"),
        ("hedge", "share", 1.0, r"hedge\.share: 1\.0 must be in \[0, 1\)"),
        ("cluster", "sim", -0.1, r"cluster\.sim: -0\.1 must be in \[0, 1\]"),
        ("run", "seed", -1, r"run\.seed: -1 must be in"),
        ("run", "seed", 2**32, r"run\.seed: 4294967296 must be in"),
    ],
)
def test_config_out_of_range_value_is_rejected(default_cfg_dict, section, key, value, message):
    default_cfg_dict[section][key] = value

    with pytest.raises(ConfigError, match=message):
        config_from_dict(default_cfg_dict)


def test_config_empty_prompt_list_is_rejected(default_cfg_dict):
    default_cfg_dict["label"]["prompts"] = []

    with pytest.raises(ConfigError, match=r"label\.prompts: must not be empty"):
        config_from_dict(default_cfg_dict)


def test_config_prompt_without_slot_is_rejected(default_cfg_dict):
    # Without "{}" every category gets the same text, so the label is a coin flip.
    default_cfg_dict["label"]["prompts"] = ["a video of {}", "a video"]

    with pytest.raises(ConfigError, match=r"label\.prompts: \['a video'\] have no"):
        config_from_dict(default_cfg_dict)


def test_config_hash_ignores_key_order_and_int_vs_float(default_cfg_dict):
    reordered = dict(reversed(list(default_cfg_dict.items())))
    reordered["pairing"] = {**reordered["pairing"], "alpha": 1}  # 1 vs 1.0 in the file

    assert config_hash(config_from_dict(reordered)) == config_hash(
        config_from_dict(default_cfg_dict)
    )


def test_config_hash_changes_when_a_value_changes(default_cfg_dict):
    before = config_hash(config_from_dict(default_cfg_dict))
    default_cfg_dict["select"]["stride"] = 0.5  # any value other than the default

    assert config_hash(config_from_dict(default_cfg_dict)) != before


# --- P3 additions: features.flow/hands/clip blocks, label.classes mapping ------------


def test_config_label_classes_loads_as_str_mapping(default_cfg_dict):
    cfg = config_from_dict(default_cfg_dict)

    assert cfg.label.classes["ski"] == "skiing"
    assert "cooking" in cfg.label.classes


def test_config_label_classes_without_cooking_is_rejected(default_cfg_dict):
    del default_cfg_dict["label"]["classes"]["cooking"]

    with pytest.raises(ConfigError, match=r"label\.classes: must include 'cooking'"):
        config_from_dict(default_cfg_dict)


def test_config_label_classes_list_instead_of_mapping_is_rejected(default_cfg_dict):
    default_cfg_dict["label"]["classes"] = ["ski", "cooking"]

    with pytest.raises(ConfigError, match=r"label\.classes: expected a non-empty mapping"):
        config_from_dict(default_cfg_dict)


def test_config_label_classes_non_string_phrase_is_rejected_with_its_key(default_cfg_dict):
    default_cfg_dict["label"]["classes"]["ski"] = 3

    with pytest.raises(ConfigError, match=r"label\.classes\.ski: expected a non-empty string"):
        config_from_dict(default_cfg_dict)


def test_config_flow_width_zero_is_rejected(default_cfg_dict):
    default_cfg_dict["features"]["flow"]["width"] = 0

    with pytest.raises(ConfigError, match=r"features\.flow\.width: 0 must be > 0"):
        config_from_dict(default_cfg_dict)


def test_config_hands_threshold_above_one_is_rejected(default_cfg_dict):
    default_cfg_dict["features"]["hands"]["min_presence_conf"] = 1.5

    with pytest.raises(ConfigError, match=r"features\.hands\.min_presence_conf: 1\.5 must be"):
        config_from_dict(default_cfg_dict)


# --- P4 additions: pairing.score_power, zeroshot ------------------------------------


def test_config_zeroshot_template_without_slot_is_rejected(default_cfg_dict):
    default_cfg_dict["zeroshot"]["verb_template"] = "a photo of hands"

    with pytest.raises(ConfigError, match=r"zeroshot\.verb_template: .* has no '\{\}' slot"):
        config_from_dict(default_cfg_dict)


def test_config_zeroshot_zero_stride_is_rejected(default_cfg_dict):
    default_cfg_dict["zeroshot"]["stride_sec"] = 0.0

    with pytest.raises(ConfigError, match=r"zeroshot\.stride_sec: 0\.0 must be > 0"):
        config_from_dict(default_cfg_dict)


def test_config_score_power_zero_is_rejected(default_cfg_dict):
    # power 0 would map every pair score to 1.0 and destroy the ranking mAP needs
    default_cfg_dict["pairing"]["score_power"] = 0.0

    with pytest.raises(ConfigError, match=r"pairing\.score_power: 0\.0 must be > 0"):
        config_from_dict(default_cfg_dict)


# --- P6 additions: importance window grid ---------------------------------------------


@pytest.mark.parametrize("key", ["len_step", "stride"])
def test_config_importance_grid_step_zero_is_rejected(default_cfg_dict, key):
    # a zero step would make the FPVSum window search loop forever / produce nothing
    default_cfg_dict["importance"][key] = 0.0

    with pytest.raises(ConfigError, match=rf"importance\.{key}: 0\.0 must be > 0"):
        config_from_dict(default_cfg_dict)


def test_config_detect_released_features_is_a_path_string(default_cfg_dict):
    cfg = config_from_dict(default_cfg_dict)
    assert cfg.detect.released_features == "weights/released_features"

    default_cfg_dict["detect"]["released_features"] = ""
    msg = r"detect\.released_features: expected a non-empty string"
    with pytest.raises(ConfigError, match=msg):
        config_from_dict(default_cfg_dict)
