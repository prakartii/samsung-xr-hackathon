"""Shared fixtures. Fixtures are built in code (CLAUDE.md §7.3 rule 9)."""

import copy
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
KIT_DIR = REPO_ROOT / "kit"


@pytest.fixture
def default_cfg_dict() -> dict:
    """A fresh, mutable copy of configs/default.yaml as a plain dict."""
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        return copy.deepcopy(yaml.safe_load(f))


@pytest.fixture(scope="session")
def no_model_config(tmp_path_factory) -> Path:
    """configs/default.yaml with the CLIP weights path pointing nowhere.

    run.sh / run_video then write the validated fallback without loading any model,
    whether or not weights/ exists on this machine (CLAUDE.md §7.3 rule 6: no real
    model in the default suite). The model path is tested with fakes instead.
    """
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["features"]["clip"]["weights"] = "weights/absent-in-tests.safetensors"
    path = tmp_path_factory.mktemp("config") / "no_models.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path
