"""tools/predict_fpvsum.py: FPVSum documents must pass the kit validator."""

import importlib.util
from pathlib import Path

from rewind.config import load_config
from rewind.schema import validate
from rewind.vocab import load_vocab
from tests.conftest import DEFAULT_CONFIG

_PATH = Path(__file__).resolve().parents[2] / "tools" / "predict_fpvsum.py"
_spec = importlib.util.spec_from_file_location("predict_fpvsum", _PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_fpvsum_document_is_valid_with_label_and_no_actions():
    # 100 s video, two selected windows (8.1 s total <= 14.5 s budget), label "ski".
    cfg = load_config(DEFAULT_CONFIG)

    doc = tool.fpvsum_document("Ski_x", 100.0, [(10.0, 14.05), (50.0, 54.05)], ("ski", 0.7), cfg)

    errors, _ = validate(doc, load_vocab(), filename_stem="Ski_x", strict=False)
    assert errors == []
    assert doc["label"]["name"] == "ski"
    assert doc["actions"] == []  # kit: FPVSum actions may be [] (mAP is EPIC-only)
    assert [h["segment"] for h in doc["highlights"]] == [[10.0, 14.05], [50.0, 54.05]]


def test_fpvsum_document_with_no_segments_still_has_a_highlight():
    # Selection can return nothing (very short video); post-processing must still
    # leave >= 1 highlight of >= 4 s (kit R11).
    cfg = load_config(DEFAULT_CONFIG)

    doc = tool.fpvsum_document("Ski_x", 60.0, [], ("ski", 0.7), cfg)

    errors, _ = validate(doc, load_vocab(), filename_stem="Ski_x", strict=False)
    assert errors == []
    assert doc["highlights"]
