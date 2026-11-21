"""tools/eval_selector_train.py: video sampling and variant parsing."""

import importlib.util
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(_TOOLS))
_spec = importlib.util.spec_from_file_location(
    "eval_selector_train", _TOOLS / "eval_selector_train.py"
)
tool = importlib.util.module_from_spec(_spec)
sys.modules["eval_selector_train"] = tool
_spec.loader.exec_module(tool)


def test_sample_train_ids_never_returns_excluded_ids():
    ids = [f"P{i:02d}_01" for i in range(30)]
    excluded = set(ids[:10])
    got = tool.sample_train_ids(ids, excluded, n=15, seed=0)
    assert len(got) == 15
    assert not set(got) & excluded


def test_sample_train_ids_is_fixed_by_the_seed_and_sorted():
    ids = [f"P{i:02d}_01" for i in range(30)]
    a = tool.sample_train_ids(ids, set(), n=10, seed=3)
    assert a == tool.sample_train_ids(list(reversed(ids)), set(), n=10, seed=3)
    assert a == sorted(a)


def test_sample_train_ids_caps_at_the_pool_size():
    assert tool.sample_train_ids(["a", "b", "c"], {"b"}, n=10, seed=0) == ["a", "c"]


def test_parse_variant_reads_name_and_yaml_values():
    name, ov = tool.parse_variant("x=select.stride=0.5;cluster.span_mode=median;a.b=[1, 2]")
    assert name == "x"
    assert ov == {"select.stride": 0.5, "cluster.span_mode": "median", "a.b": [1, 2]}


def test_parse_variant_without_overrides_is_the_base_config():
    assert tool.parse_variant("base=") == ("base", {})


def test_dev_val_ids_are_disjoint_and_dev_matches_the_plain_sample():
    ids = [f"P{i:02d}_{j:02d}" for i in range(20) for j in range(10)]  # 200 ids
    excluded = set(ids[:20])
    dev, val = tool.dev_val_ids(ids, excluded, n=50, seed=0)
    assert len(dev) == 50 and len(val) == 50
    assert not set(dev) & set(val)
    assert not (set(dev) | set(val)) & excluded
    assert dev == tool.sample_train_ids(ids, excluded, n=50, seed=0)
