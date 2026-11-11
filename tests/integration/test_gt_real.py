"""Real annotation files (marker `data`; run tools/fetch_annotations.sh first)."""

import csv

import numpy as np
import pytest

from rewind_eval.gt_epic import load_epic_actions
from rewind_eval.gt_fpvsum import load_fpvsum
from tests.conftest import KIT_DIR, REPO_ROOT

ANN = REPO_ROOT / "data" / "annotations"
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (ANN / "epic" / "EPIC_100_validation.csv").exists()
        or not (ANN / "fpvsum" / "gt_data.mat").exists(),
        reason="run tools/fetch_annotations.sh",
    ),
]


def manifest(dataset):
    with open(KIT_DIR / "manifest.csv", encoding="utf-8", newline="") as f:
        return {
            r["video_id"]: float(r["duration_sec"])
            for r in csv.DictReader(f)
            if r["dataset"] == dataset
        }


def test_every_manifest_epic_video_has_validation_actions_in_kit_vocabulary():
    import json

    classes = json.loads((KIT_DIR / "supported_classes.json").read_text(encoding="utf-8"))
    gt = load_epic_actions(
        ANN / "epic" / "EPIC_100_validation.csv",
        ANN / "epic" / "EPIC_100_verb_classes.csv",
        ANN / "epic" / "EPIC_100_noun_classes.csv",
    )
    verbs, nouns = set(classes["verbs"]), set(classes["nouns"])

    assert set(manifest("epic_kitchens_100")) <= set(gt)
    assert all(v in verbs and n in nouns for acts in gt.values() for _, v, n in acts)


def test_every_manifest_fpvsum_video_has_ground_truth():
    dur = manifest("fpvsum")

    gt = load_fpvsum(ANN / "fpvsum" / "gt_data.mat", dur)

    assert set(gt) == set(dur)  # all 49, including the Jumping__ name quirk
    fractions = [float(np.mean(g.highlight)) for g in gt.values()]
    assert min(fractions) > 0.10 and max(fractions) < 0.30


def test_every_fpvsum_label_is_a_kit_label():
    import json

    labels = set(
        json.loads((KIT_DIR / "supported_classes.json").read_text(encoding="utf-8"))["labels"]
    )

    gt = load_fpvsum(ANN / "fpvsum" / "gt_data.mat", manifest("fpvsum"))

    assert {g.label for g in gt.values()} <= labels
