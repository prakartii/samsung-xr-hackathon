"""tools/make_submission.py: the P9 package (kit §6, ARCHITECTURE.md §13).

The package is what the organizers re-run, so these tests check what it contains and
that its own run.sh works with nothing from the repo around it.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from rewind.config import load_config
from rewind.fallback import fallback_result
from rewind.output import dumps, to_document
from rewind.pipeline import run_video
from rewind_eval.validate import read_manifest
from tests.conftest import DEFAULT_CONFIG, REPO_ROOT

from .test_run_sh import make_video

_spec = importlib.util.spec_from_file_location(
    "make_submission", REPO_ROOT / "tools" / "make_submission.py"
)
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)

CFG = load_config(DEFAULT_CONFIG)
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def write_outputs(out: Path, durations: dict[str, float], team_id: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for vid, d in durations.items():
        r = fallback_result(team_id, vid, d, budget_frac=CFG.select.budget_frac,
                            merge_gap=CFG.post.merge_gap,
                            short_video_mode=CFG.post.short_video_mode)  # fmt: skip
        (out / f"{vid}.json").write_text(dumps(to_document(r)), encoding="utf-8")
    return out


MANIFEST = {"vid_a": 60.0, "vid_b": 120.0}


def no_models(*_):
    raise RuntimeError("no models in the default suite")


@pytest.mark.parametrize("team_id", ["TBD_Q11", "tbd", "", "  "])
def test_team_id_problems_placeholder_is_rejected(team_id):
    assert len(ms.team_id_problems(team_id)) == 1


def test_team_id_problems_real_id_passes():
    assert ms.team_id_problems("T07") == []


def test_weight_paths_default_config_lists_every_weights_entry():
    # configs/default.yaml names exactly these six paths under weights/.
    assert ms.weight_paths(CFG) == [
        "weights/actionformer_epic_noun.pth.tar",
        "weights/actionformer_epic_verb.pth.tar",
        "weights/clip_vit_b16_openai.safetensors",
        "weights/hand_landmarker.task",
        "weights/released_features",
        "weights/vn_prior.npy",
    ]


def test_missing_weights_reports_absent_file_and_empty_directory(tmp_path):
    (tmp_path / "weights" / "feats").mkdir(parents=True)  # empty: nothing to find
    (tmp_path / "weights" / "a.bin").write_bytes(b"x")
    got = ms.missing_weights(tmp_path, ["weights/a.bin", "weights/b.bin", "weights/feats"])
    assert len(got) == 2
    assert "weights/b.bin" in got[0] and "weights/feats" in got[1]


def test_id_mentions_counts_whole_ids_only(tmp_path):
    # Line 1 has the ID as a token; lines 2-3 only contain it inside longer tokens;
    # line 4 has an ID that itself contains '-' and '_'.
    (tmp_path / "f.py").write_text(
        "# seen on P01_11 first\nP01_110\nxP01_11\nSki_-Z5a4UQmZLs.mp4\n", encoding="utf-8"
    )
    got = ms.id_mentions(tmp_path, [Path("f.py")], ["P01_11", "Ski_-Z5a4UQmZLs"])
    assert got == ["f.py:1: video ID P01_11", "f.py:4: video ID Ski_-Z5a4UQmZLs"]


def test_shipped_files_mention_no_manifest_video_id():
    # Kit §6: automated scans for hardcoded video IDs disqualify, so nothing the package
    # ships (code, config, README, Dockerfile) names a manifest video, even in a comment.
    ids = read_manifest(REPO_ROOT / "kit" / "manifest.csv")
    assert ms.id_mentions(REPO_ROOT, ms.shipped_files(REPO_ROOT), ids) == []


def test_output_problems_complete_valid_directory_passes(tmp_path):
    out = write_outputs(tmp_path / "out", MANIFEST, "T07")
    assert ms.output_problems(out, MANIFEST, "T07") == []


def test_output_problems_missing_manifest_video_is_reported(tmp_path):
    out = write_outputs(tmp_path / "out", {"vid_a": 60.0}, "T07")
    got = ms.output_problems(out, MANIFEST, "T07")
    assert len(got) == 1 and "vid_b" in got[0]


def test_output_problems_team_id_other_than_config_is_reported(tmp_path):
    # Outputs made before the team ID was set would not match a re-run of run.sh.
    out = write_outputs(tmp_path / "out", MANIFEST, "TBD_Q11")
    got = ms.output_problems(out, MANIFEST, "T07")
    assert len(got) == 2 and all("team_id" in g for g in got)


def test_output_problems_invalid_file_is_reported(tmp_path):
    out = write_outputs(tmp_path / "out", MANIFEST, "T07")
    doc = json.loads((out / "vid_a.json").read_text(encoding="utf-8"))
    doc["highlights"] = []
    (out / "vid_a.json").write_text(json.dumps(doc), encoding="utf-8")
    got = ms.output_problems(out, MANIFEST, "T07")
    assert got and all(g.startswith("vid_a.json") for g in got)


@pytest.fixture
def repo(tmp_path) -> Path:
    """A copy of the repo's code and docs (no data, weights or git) with stale bytecode."""
    root = tmp_path / "repo"
    skip = (".git", ".venv", "data", "cache", "weights", "tests", "results", "__pycache__")
    shutil.copytree(REPO_ROOT, root, ignore=shutil.ignore_patterns(*skip))
    (root / "code" / "rewind" / "__pycache__").mkdir()
    (root / "code" / "rewind" / "__pycache__" / "cli.cpython-311.pyc").write_bytes(b"\0")
    return root


def test_assemble_ships_only_the_kit_layout(repo, tmp_path):
    out = write_outputs(tmp_path / "out", MANIFEST, "T07")
    pkg = tmp_path / "dist" / "T07"
    ms.assemble(repo, pkg, out, MANIFEST, weights=[])

    top = sorted(p.name for p in pkg.iterdir())
    assert top == [".dockerignore", "Dockerfile", "README.md", "code", "configs", "outputs",
                   "requirements.txt", "run.sh"]  # fmt: skip
    assert sorted(p.name for p in (pkg / "code").iterdir()) == [
        "rewind", "rewind_eval", "third_party",
    ]  # fmt: skip
    assert sorted(p.name for p in (pkg / "configs").iterdir()) == ["default.yaml"]
    assert sorted(p.name for p in (pkg / "outputs").iterdir()) == ["vid_a.json", "vid_b.json"]
    assert not list(pkg.rglob("__pycache__"))
    assert (pkg / "run.sh").stat().st_mode & 0o111


def test_assemble_links_weights_into_the_package(repo, tmp_path):
    (repo / "weights" / "feats").mkdir(parents=True)
    (repo / "weights" / "feats" / "x.npz").write_bytes(b"12")
    (repo / "weights" / "m.bin").write_bytes(b"3")
    out = write_outputs(tmp_path / "out", MANIFEST, "T07")
    pkg = tmp_path / "dist" / "T07"
    files = ms.weight_files(repo, ["weights/feats", "weights/m.bin", "weights/absent"])
    ms.assemble(repo, pkg, out, MANIFEST, files)

    assert (pkg / "weights" / "feats" / "x.npz").read_bytes() == b"12"
    assert (pkg / "weights" / "m.bin").read_bytes() == b"3"
    assert sorted(p.name for p in (pkg / "weights").iterdir()) == ["feats", "m.bin"]


@needs_ffmpeg
def test_packaged_run_sh_reproduces_its_output(tmp_path):
    # The package has no weights here, so its run.sh writes the fallback; the expected
    # output is the same fallback made by the repo's pipeline. Byte-equal means the
    # package runs on its own code and config, with nothing from the repo around it.
    video = make_video(tmp_path / "vid_a.mp4", 12)
    out = tmp_path / "out"
    run_video(video, out / "vid_a.json", CFG, team_id=CFG.team.team_id,
              analyze=no_models)  # fmt: skip
    pkg = tmp_path / "dist" / "T07"
    ms.assemble(REPO_ROOT, pkg, out, {"vid_a": 12.0}, weights=[])

    assert ms.rerun_problems(pkg, video, ms.sys.executable, timeout=300) == []


def fake_package(pkg: Path, script: str) -> Path:
    # A stand-in run.sh: copies the packaged JSON to <out> and runs `script` first.
    (pkg / "outputs").mkdir(parents=True)
    (pkg / "outputs" / "v.json").write_text("{}", encoding="utf-8")
    run = pkg / "run.sh"
    run.write_text(f'#!/usr/bin/env bash\ncd "$(dirname "$0")"\n{script}\n'
                   'cp outputs/v.json "$2"\n', encoding="utf-8")  # fmt: skip
    run.chmod(0o755)
    return pkg


def test_rerun_problems_removes_the_cache_the_run_created(tmp_path):
    # run.sh caches features under <package>/cache; the package must stay as assembled.
    pkg = fake_package(tmp_path / "T07", "mkdir -p cache/flow && touch cache/flow/x.npz")
    assert ms.rerun_problems(pkg, tmp_path / "v.mp4", "python3", timeout=30) == []
    assert not (pkg / "cache").exists()


def test_rerun_problems_existing_cache_is_refused(tmp_path):
    # A warm cache would hide a run that cannot recompute features from the MP4.
    pkg = fake_package(tmp_path / "T07", "")
    (pkg / "cache").mkdir()
    got = ms.rerun_problems(pkg, tmp_path / "v.mp4", "python3", timeout=30)
    assert len(got) == 1 and "empty cache" in got[0]


def test_rerun_problems_nonzero_exit_is_reported(tmp_path):
    pkg = fake_package(tmp_path / "T07", "exit 2")
    got = ms.rerun_problems(pkg, tmp_path / "v.mp4", "python3", timeout=30)
    assert len(got) == 1 and "exited 2" in got[0]


@needs_ffmpeg
def test_packaged_run_sh_output_differing_from_packaged_json_is_reported(tmp_path):
    video = make_video(tmp_path / "vid_a.mp4", 12)
    out = tmp_path / "out"
    run_video(video, out / "vid_a.json", CFG, team_id="SOMEONE_ELSE",
              analyze=no_models)  # fmt: skip
    pkg = tmp_path / "dist" / "T07"
    ms.assemble(REPO_ROOT, pkg, out, {"vid_a": 12.0}, weights=[])

    got = ms.rerun_problems(pkg, video, ms.sys.executable, timeout=300)
    assert len(got) == 1 and "differs" in got[0]


def test_main_with_placeholder_team_id_writes_nothing(tmp_path, capsys):
    # configs/default.yaml still has TBD_Q11 (Q11 unanswered): no package at all.
    out = write_outputs(tmp_path / "out", read_manifest(REPO_ROOT / "kit" / "manifest.csv"),
                        CFG.team.team_id)  # fmt: skip
    rc = ms.main(["--outputs", str(out), "--dest", str(tmp_path / "dist")])

    assert rc == 1
    assert not (tmp_path / "dist").exists()
    assert "placeholder" in capsys.readouterr().err
