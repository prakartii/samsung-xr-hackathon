"""rewind_eval.validate.validate_dir: cross-file rules on a directory of outputs."""

import copy
import json

from rewind_eval.validate import validate_dir
from tests.unit.test_schema import BASE


def write(dir_, video_id, **changes):
    doc = copy.deepcopy(BASE)
    doc.update(video_id=video_id, **changes)
    (dir_ / f"{video_id}.json").write_text(json.dumps(doc), encoding="utf-8")


def test_validate_dir_complete_and_consistent_has_no_errors(tmp_path):
    write(tmp_path, "A")
    write(tmp_path, "B")

    errors, warnings = validate_dir(tmp_path, {"A": 100.0, "B": 100.2}, strict=True)

    assert (errors, warnings) == ({}, {})


def test_validate_dir_missing_manifest_video_is_error(tmp_path):
    write(tmp_path, "A")

    errors, _ = validate_dir(tmp_path, {"A": 100.0, "B": 100.0}, strict=True)

    assert errors == {"<dir>": ["1 manifest video(s) have no output: B"]}


def test_validate_dir_team_id_mismatch_is_error(tmp_path):
    write(tmp_path, "A")
    write(tmp_path, "B", team_id="T08")

    errors, _ = validate_dir(tmp_path, {"A": 100.0, "B": 100.0}, strict=True)

    assert errors == {"<dir>": ["team_id differs across files: ['T07', 'T08']"]}


def test_validate_dir_file_not_in_manifest_is_warning(tmp_path):
    write(tmp_path, "A")
    write(tmp_path, "Z")

    errors, warnings = validate_dir(tmp_path, {"A": 100.0}, strict=True)

    assert errors == {}
    assert warnings == {"<dir>": ["1 file(s) not in the manifest: Z"]}


def test_validate_dir_duration_checked_against_manifest(tmp_path):
    write(tmp_path, "A")  # duration_sec 100.0 vs manifest 101.0: off by 1.0 > 0.5

    errors, _ = validate_dir(tmp_path, {"A": 101.0}, strict=True)

    assert list(errors) == ["A.json"]
    assert errors["A.json"][0].startswith("R3 ")


def test_validate_dir_invalid_json_is_error(tmp_path):
    (tmp_path / "A.json").write_text("{not json", encoding="utf-8")

    errors, _ = validate_dir(tmp_path, {"A": 100.0}, strict=True)

    assert errors["A.json"][0].startswith("R1 <root>: not valid JSON")
