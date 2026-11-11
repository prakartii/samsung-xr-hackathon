"""EPIC action chain on real released features (tier: data). Run with `pytest -m data`.

cached SlowFast features -> ActionFormer verb + noun (CPU) -> pairing -> actions[].
Needs: weights (tools/fetch_weights.sh), weights/vn_prior.npy
(tools/build_vn_prior.py), data/videos/P01_12.mp4 and its released features imported
(tools/import_released_features.py). Skips when any is missing.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from rewind.config import load_config
from rewind.constants import MAX_ACTIONS
from rewind.detect.actionformer import ActionFormerDetector, ActionFormerParams
from rewind.detect.pairing import PairingParams, VNPrior, pair_proposals, proposals_to_actions
from rewind.features.cache import content_fingerprint
from rewind.features.slowfast import slowfast_cache
from rewind.video_io import probe_video

pytestmark = pytest.mark.data

REPO = Path(__file__).resolve().parents[2]
VIDEO = REPO / "data" / "videos" / "P01_12.mp4"
CFG = load_config(REPO / "configs" / "default.yaml")
KIT = json.loads((REPO / "kit" / "supported_classes.json").read_text(encoding="utf-8"))
AF = REPO / "code" / "third_party" / "actionformer" / "configs"


@pytest.fixture(scope="module")
def chain():
    needed = [
        REPO / "weights" / "actionformer_epic_verb.pth.tar",
        REPO / "weights" / "actionformer_epic_noun.pth.tar",
        REPO / "weights" / "vn_prior.npy",
        VIDEO,
    ]
    if missing := [str(p) for p in needed if not p.exists()]:
        pytest.skip(f"missing: {missing}")
    entry = slowfast_cache(REPO / "cache").get(content_fingerprint(VIDEO))
    if entry is None:
        pytest.skip("P01_12 features not imported: run tools/import_released_features.py")
    duration = probe_video(VIDEO).duration_sec

    def detector(kind: str) -> ActionFormerDetector:
        params = ActionFormerParams(
            str(AF / f"epic_slowfast_{kind}.yaml"), f"weights/actionformer_epic_{kind}.pth.tar"
        )
        return ActionFormerDetector(params, KIT[f"{kind}s"], REPO)

    verbs = detector("verb").detect(entry["feats"], duration)
    nouns = detector("noun").detect(entry["feats"], duration)
    prior = VNPrior.from_counts(
        KIT["verbs"], KIT["nouns"], np.load(REPO / "weights" / "vn_prior.npy"),
        CFG.pairing.eps,
    )  # fmt: skip
    proposals = pair_proposals(
        verbs,
        nouns,
        prior,
        set(KIT["verbs"]),
        set(KIT["nouns"]),
        PairingParams(**asdict(CFG.pairing)),
    )
    actions = proposals_to_actions(proposals, CFG.actions.max_per_video, CFG.actions.min_score)
    return duration, verbs, nouns, actions


def test_epic_detectors_emit_detections_inside_the_video(chain):
    duration, verbs, nouns, _ = chain
    assert verbs and nouns
    for (s, e), _, p in verbs + nouns:
        assert 0.0 <= s < e <= duration
        assert 0.0 <= p <= 1.0


def test_epic_actions_are_valid_kit_actions(chain):
    duration, _, _, actions = chain
    assert 1 <= len(actions) <= min(CFG.actions.max_per_video, MAX_ACTIONS)
    verbs, nouns = set(KIT["verbs"]), set(KIT["nouns"])
    for (s, e), v, n, score in actions:
        assert 0.0 <= s < e <= duration + 0.5  # kit R6
        assert v in verbs and n in nouns  # kit R7
        assert 0.0 <= score <= 1.0  # kit R8
    scores = [a[3] for a in actions]
    assert scores == sorted(scores, reverse=True)
