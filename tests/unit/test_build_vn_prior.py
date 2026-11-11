"""Counting for tools/build_vn_prior.py on a 5-row CSV-like fixture."""

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[2] / "tools" / "build_vn_prior.py"
_spec = importlib.util.spec_from_file_location("build_vn_prior", _PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_count_pairs_hand_computed():
    rows = [
        {"verb_class": "0", "noun_class": "1"},
        {"verb_class": "0", "noun_class": "1"},
        {"verb_class": "0", "noun_class": "0"},
        {"verb_class": "1", "noun_class": "2"},
        {"verb_class": "2", "noun_class": "2"},
    ]
    # (0,1) x2, (0,0) x1, (1,2) x1, (2,2) x1; everything else 0
    want = np.array([[1, 2, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
    np.testing.assert_array_equal(tool.count_pairs(rows, 3, 3), want)


def test_count_pairs_last_class_other_stays_zero():
    # Kit `other` is index n-1 and never appears in EPIC annotations.
    counts = tool.count_pairs([{"verb_class": "0", "noun_class": "0"}], 2, 2)
    assert counts[-1].sum() == 0 and counts[:, -1].sum() == 0
