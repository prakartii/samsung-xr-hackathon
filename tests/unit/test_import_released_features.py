"""Alignment arithmetic of tools/import_released_features.py."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "import_released_features.py"
_spec = importlib.util.spec_from_file_location("import_released_features", _PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_expected_feature_count_hand_computed():
    # 10 s -> 300 frames at 30 fps -> (300 - 32) // 16 + 1 = 268 // 16 + 1 = 16 + 1 = 17
    assert tool.expected_feature_count(10.0) == 17


def test_expected_feature_count_shorter_than_one_clip_is_zero():
    # 1 s = 30 frames < 32 -> no full clip
    assert tool.expected_feature_count(1.0) == 0


def test_alignment_gap_in_seconds():
    # 10 s expects 17 features; 20 features -> 3 steps * 16/30 s = 1.6 s
    assert tool.alignment_gap_sec(20, 10.0) == pytest.approx(1.6, abs=1e-12)
    assert tool.alignment_gap_sec(17, 10.0) == pytest.approx(0.0, abs=1e-12)


# --- fingerprints from the Kaggle job's logs (videos not on this machine) -------------


def _write_log(path, records):
    import json

    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_fingerprint_logs_map_ids_and_skip_failures(tmp_path):
    a = _write_log(
        tmp_path / "shard_0.jsonl",
        [
            {"video_id": "P01_11", "fingerprint": "aa", "duration": 561.5, "seconds": 9.0},
            {"video_id": "P08_09", "error": "DecodeError(...)"},
        ],
    )
    b = _write_log(
        tmp_path / "shard_0_retry.jsonl",
        [{"video_id": "P08_09", "fingerprint": "bb", "duration": 620.5, "seconds": 7.0}],
    )
    assert tool.load_fingerprint_logs([a, b]) == {"P01_11": ("aa", 561.5), "P08_09": ("bb", 620.5)}


def test_fingerprint_logs_same_id_two_fingerprints_is_an_error(tmp_path):
    # Two different files under one id: importing either could attach features to the
    # wrong video, so refuse.
    log = _write_log(
        tmp_path / "shard_0.jsonl",
        [
            {"video_id": "P01_11", "fingerprint": "aa", "duration": 561.5},
            {"video_id": "P01_11", "fingerprint": "zz", "duration": 561.5},
        ],
    )
    with pytest.raises(ValueError):
        tool.load_fingerprint_logs([log])


def test_ids_to_import_adds_logged_non_manifest_videos():
    # Manifest P01_11, P02_01; logs P02_01 (also in manifest) and tune P12_05, P09_104.
    # -> manifest order first, then the extra logged ids sorted, no duplicates.
    logged = {"P02_01": ("fa", 1.0), "P12_05": ("fb", 2.0), "P09_104": ("fc", 3.0)}
    assert tool.ids_to_import(["P01_11", "P02_01"], logged) == [
        "P01_11",
        "P02_01",
        "P09_104",
        "P12_05",
    ]
