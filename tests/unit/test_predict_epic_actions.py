"""tools/predict_epic_actions.py: the document it writes must pass the kit validator."""

import importlib.util
from pathlib import Path

from rewind.config import load_config
from rewind.schema import validate
from rewind.vocab import load_vocab
from tests.conftest import DEFAULT_CONFIG

_PATH = Path(__file__).resolve().parents[2] / "tools" / "predict_epic_actions.py"
_spec = importlib.util.spec_from_file_location("predict_epic_actions", _PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_epic_document_is_valid_and_keeps_the_actions():
    cfg = load_config(DEFAULT_CONFIG)
    actions = [((1.0, 3.5), "take", "knife", 0.9), ((10.0, 12.0), "open", "fridge", 0.4)]

    doc = tool.epic_document("P01_11", 120.0, actions, cfg)

    errors, _ = validate(doc, load_vocab(), filename_stem="P01_11", strict=False)
    assert errors == []
    assert doc["label"]["name"] == "cooking"
    pairs = [(a["verb"], a["noun"]) for a in doc["actions"]]
    assert pairs == [("take", "knife"), ("open", "fridge")]


def test_epic_document_with_no_actions_is_still_valid():
    # A video where pairing finds nothing must still score (empty actions are allowed).
    cfg = load_config(DEFAULT_CONFIG)

    doc = tool.epic_document("P01_11", 120.0, [], cfg)

    errors, _ = validate(doc, load_vocab(), filename_stem="P01_11", strict=False)
    assert errors == []
    assert doc["actions"] == []
