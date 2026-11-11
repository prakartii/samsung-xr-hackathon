"""GT loaders and splits on small fixtures built in code (no files from data/)."""

import importlib.util

import numpy as np
import pytest
import scipy.io

from rewind_eval.gt_epic import load_epic_actions, parse_timestamp
from rewind_eval.gt_fpvsum import load_fpvsum, name_key
from rewind_eval.splits import load_splits
from tests.conftest import KIT_DIR, REPO_ROOT

HEADER = (
    "narration_id,participant_id,video_id,narration_timestamp,start_timestamp,"
    "stop_timestamp,start_frame,stop_frame,narration,verb,verb_class,noun,noun_class,"
    "all_nouns,all_noun_classes\n"
)


def row(vid, start, stop, verb, verb_class, noun, noun_class):
    return (
        f"x,P01,{vid},{start},{start},{stop},1,2,n,{verb},{verb_class},{noun},{noun_class},[],[]\n"
    )


@pytest.fixture
def epic_files(tmp_path):
    ann = tmp_path / "ann.csv"
    ann.write_text(
        HEADER
        + row("P01_11", "00:00:05.00", "00:00:07.50", "put-down", 1, "plate", 0)
        + row("P01_11", "00:00:01.00", "00:00:02.00", "grab", 0, "knife", 1)
        + row("P01_11", "00:00:03.00", "00:00:03.00", "take", 0, "plate", 0)  # zero length
        + row("P01_12", "00:01:00.00", "00:01:02.50", "take", 0, "knife", 1),
        encoding="utf-8",
    )
    verbs = tmp_path / "verbs.csv"
    verbs.write_text("id,key,instances,category\n0,take,[],x\n1,put,[],x\n", encoding="utf-8")
    nouns = tmp_path / "nouns.csv"
    nouns.write_text("id,key,instances,category\n0,plate,[],x\n1,knife,[],x\n", encoding="utf-8")
    return ann, verbs, nouns


def test_parse_timestamp_hours_minutes_seconds():
    # 1 h + 2 min + 3.25 s = 3723.25 s
    assert parse_timestamp("01:02:03.25") == pytest.approx(3723.25, abs=1e-9)


def test_load_epic_actions_maps_class_ids_to_keys_not_raw_verbs(epic_files):
    gt = load_epic_actions(*epic_files)

    # "put-down" (class 1) → "put"; "grab" (class 0) → "take"; sorted by start.
    assert gt["P01_11"] == [((1.0, 2.0), "take", "knife"), ((5.0, 7.5), "put", "plate")]


def test_load_epic_actions_skips_zero_length_and_groups_by_video(epic_files):
    gt = load_epic_actions(*epic_files)

    assert sorted(gt) == ["P01_11", "P01_12"]
    assert gt["P01_12"] == [((60.0, 62.5), "take", "knife")]


def test_name_key_ignores_leading_underscores_of_the_youtube_id():
    assert (
        name_key("Jumping__mU-p2wuk90")
        == name_key("Jumping_mU-p2wuk90")
        == (
            "jumping",
            "mU-p2wuk90",
        )
    )


def test_load_fpvsum_matches_names_and_spreads_frames_over_duration(tmp_path):
    mat = tmp_path / "gt.mat"
    records = np.empty(2, dtype=object)
    records[0] = {"Name": "Jumping_mU-p2wuk90", "Score": np.array([0.0, 0.5, 1.0, 0.25]),
                  "Binary_score": np.array([0, 0, 1, 0], dtype=np.uint8)}  # fmt: skip
    records[1] = {"Name": "RockClimbing_abc", "Score": np.zeros(3),
                  "Binary_score": np.zeros(3, dtype=np.uint8)}  # fmt: skip
    scipy.io.savemat(mat, {"gt_data": records})

    gt = load_fpvsum(mat, {"Jumping__mU-p2wuk90": 8.0, "Bike_missing": 10.0})

    assert list(gt) == ["Jumping__mU-p2wuk90"]
    g = gt["Jumping__mU-p2wuk90"]
    # 4 frames over 8 s → frame i at 2i s.
    assert g.times.tolist() == pytest.approx([0.0, 2.0, 4.0, 6.0], abs=1e-12)
    assert g.highlight.tolist() == [False, False, True, False]
    assert g.label == "jumping"


def _make_splits_module():
    spec = importlib.util.spec_from_file_location(
        "make_splits", REPO_ROOT / "tools" / "make_splits.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pick_tune_epic_takes_nearest_duration_per_quantile():
    mod = _make_splits_module()
    mod.N_TUNE_EPIC = 2
    # val quantiles of [10, 20, 30, 40] at (i + .5) · 4 / 2 → indices 1, 3 → 20, 40.
    # Nearest train: 21 (A) for 20; 39 (C) beats 45 (D) for 40.
    picked = mod.pick_tune_epic([10, 20, 30, 40], {"A": 21.0, "B": 5.0, "C": 39.0, "D": 45.0})

    assert picked == ["A", "C"]


def test_fpvsum_folds_round_robin_within_categories():
    mod = _make_splits_module()
    ids = ["Bike_a", "Bike_b", "Ski_a", "Ski_b", "Ski_c", "Boxing_a"]

    folds = mod.fpvsum_folds(ids)

    # Sorted by category then id: Bike_a 0, Bike_b 1, Boxing_a 2, Ski_a 3, Ski_b 4, Ski_c 0.
    assert folds == {"Bike_a": 0, "Bike_b": 1, "Boxing_a": 2, "Ski_a": 3, "Ski_b": 4, "Ski_c": 0}


def test_committed_splits_are_consistent_with_the_manifest():
    import csv

    s = load_splits()
    with open(KIT_DIR / "manifest.csv", encoding="utf-8", newline="") as f:
        manifest = {r["video_id"]: r["dataset"] for r in csv.DictReader(f)}

    assert len(s.tune_epic) == 20
    assert not set(s.tune_epic) & set(manifest)  # tune never uses val/report videos
    assert set(s.report_epic) == {v for v, d in manifest.items() if d == "epic_kitchens_100"}
    assert set(s.report_fpvsum) == {v for v, d in manifest.items() if d == "fpvsum"}
    assert set(s.fpvsum_folds) == set(s.report_fpvsum)
