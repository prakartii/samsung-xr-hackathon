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
