"""tools/make_submission.py: the P9 package (kit §6, ARCHITECTURE.md §13).

The package is what the organizers re-run, so these tests check what it contains, that
its own run.sh runs on its own code, and that every refusal rule fires.
"""

import importlib.util
import json
import shutil
import subprocess
from dataclasses import replace
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
MANIFEST = {"vid_a": 60.0, "vid_b": 120.0}
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def write_outputs(
    out: Path, durations: dict[str, float], team_id: str, label: str = "bike"
) -> Path:
    """Valid outputs: fallback highlights with a model label (fallback label: 'other')."""
    out.mkdir(parents=True, exist_ok=True)
    for vid, d in durations.items():
        r = fallback_result(team_id, vid, d, budget_frac=CFG.select.budget_frac,
                            merge_gap=CFG.post.merge_gap,
                            short_video_mode=CFG.post.short_video_mode)  # fmt: skip
        if label != "other":
            r = replace(r, label=(label, 0.9))
        (out / f"{vid}.json").write_text(dumps(to_document(r)), encoding="utf-8")
    return out


def no_models(*_):
    raise RuntimeError("no models in the default suite")


@pytest.fixture
def repo(tmp_path) -> Path:
    """The shipped parts of the repo (plus kit/) with a stale bytecode file."""
    root = tmp_path / "repo"
    for rel in [*ms.SHIPPED, "kit"]:
        src, dst = REPO_ROOT / rel, root / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    (root / "code" / "rewind" / "__pycache__").mkdir()
    (root / "code" / "rewind" / "__pycache__" / "cli.cpython-311.pyc").write_bytes(b"\0")
    return root


# --- team ID ---------------------------------------------------------------------------


@pytest.mark.parametrize("team_id", ["TBD_Q11", "tbd", "", "  "])
def test_team_id_problems_placeholder_is_rejected(team_id):
    assert len(ms.team_id_problems(team_id)) == 1


def test_team_id_problems_real_id_passes():
    assert ms.team_id_problems("T07") == []


# --- weights ---------------------------------------------------------------------------


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


# --- ID scan ---------------------------------------------------------------------------


def test_id_mentions_matches_ids_as_case_insensitive_substrings(tmp_path):
    # Conservative like any scanner might be: plain, inside a longer token, after '-',
    # in lower case, and an ID that itself contains '-'. Line 3 has none.
    (tmp_path / "f.py").write_text(
        "# seen on P01_11 first\ncache_P01_110\nP01_1 only\nx-p01_11\nSki_-Z5a4UQmZLs.mp4\n",
        encoding="utf-8",
    )
    got = ms.id_mentions(tmp_path, [Path("f.py")], ["P01_11", "Ski_-Z5a4UQmZLs"])
    assert got == [
        "f.py:1: video ID P01_11",
        "f.py:2: video ID P01_11",
        "f.py:4: video ID p01_11",
        "f.py:5: video ID Ski_-Z5a4UQmZLs",
    ]


def test_id_mentions_skips_binary_files(tmp_path):
    (tmp_path / "w.bin").write_bytes(b"\xff\xfeP01_11\x00")
    assert ms.id_mentions(tmp_path, [Path("w.bin")], ["P01_11"]) == []


def test_shipped_files_mention_no_manifest_video_id():
    # Kit §6: automated scans for hardcoded video IDs disqualify, so nothing the package
    # ships (code, config, README, Dockerfile) names a manifest video, even in a comment.
    ids = read_manifest(REPO_ROOT / "kit" / "manifest.csv")
    assert ms.id_mentions(REPO_ROOT, ms.shipped_files(REPO_ROOT), ids) == []


# --- outputs ---------------------------------------------------------------------------


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


def test_output_problems_fallback_output_is_reported(tmp_path):
    # Label 'other' only comes from the fallback: the models did not run for vid_b.
    out = write_outputs(tmp_path / "out", MANIFEST, "T07")
    write_outputs(out, {"vid_b": 120.0}, "T07", label="other")
    got = ms.output_problems(out, MANIFEST, "T07")
    assert len(got) == 1 and got[0].startswith("vid_b.json") and "fallback" in got[0]


def test_output_problems_invalid_file_is_reported(tmp_path):
    out = write_outputs(tmp_path / "out", MANIFEST, "T07")
    doc = json.loads((out / "vid_a.json").read_text(encoding="utf-8"))
    doc["highlights"] = []
    (out / "vid_a.json").write_text(json.dumps(doc), encoding="utf-8")
    got = ms.output_problems(out, MANIFEST, "T07")
    assert got and all(g.startswith("vid_a.json") for g in got)


# --- working tree ----------------------------------------------------------------------


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    *args], check=True, capture_output=True)  # fmt: skip


def test_dirty_paths_reports_edits_and_untracked_files_only_under_shipped_paths(tmp_path):
    (tmp_path / "code" / "rewind").mkdir(parents=True)
    (tmp_path / "run.sh").write_text("a\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("a\n", encoding="utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    assert ms.dirty_paths(tmp_path) == []

    (tmp_path / "run.sh").write_text("b\n", encoding="utf-8")  # edited, shipped
    (tmp_path / "code" / "rewind" / "scratch.py").write_text("", encoding="utf-8")  # new
    (tmp_path / "notes.txt").write_text("b\n", encoding="utf-8")  # edited, not shipped
    assert ms.dirty_paths(tmp_path) == [
        "uncommitted change: run.sh",
        "uncommitted change: code/rewind/scratch.py",
    ]


# --- assembling ------------------------------------------------------------------------


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


def test_assemble_copies_weights_so_later_repo_edits_do_not_change_the_package(repo, tmp_path):
    (repo / "weights" / "feats").mkdir(parents=True)
    (repo / "weights" / "feats" / "x.npz").write_bytes(b"12")
    (repo / "weights" / "m.bin").write_bytes(b"3")
    out = write_outputs(tmp_path / "out", MANIFEST, "T07")
    pkg = tmp_path / "dist" / "T07"
    files = ms.weight_files(repo, ["weights/feats", "weights/m.bin", "weights/absent"])
    ms.assemble(repo, pkg, out, MANIFEST, files)
    with open(repo / "weights" / "m.bin", "wb") as f:  # np.save-style in-place rewrite
        f.write(b"rebuilt")

    assert (pkg / "weights" / "feats" / "x.npz").read_bytes() == b"12"
    assert (pkg / "weights" / "m.bin").read_bytes() == b"3"
    assert sorted(p.name for p in (pkg / "weights").iterdir()) == ["feats", "m.bin"]


# --- re-running the package ------------------------------------------------------------


def fake_package(pkg: Path, script: str) -> Path:
    """A stand-in run.sh: runs `script`, then copies the packaged JSON to <out>."""
    (pkg / "outputs").mkdir(parents=True)
    (pkg / "outputs" / "v.json").write_text("{}", encoding="utf-8")
    run = pkg / "run.sh"
    run.write_text(f'#!/usr/bin/env bash\nROOT="$(dirname "$0")"\n{script}\n'
                   'cp "$ROOT/outputs/v.json" "$2"\n', encoding="utf-8")  # fmt: skip
    run.chmod(0o755)
    return pkg


def rerun(pkg: Path, tmp_path: Path, timeout: float = 30) -> list[str]:
    return ms.rerun_problems(pkg, tmp_path / "v.mp4", "python3", timeout=timeout)


def test_rerun_problems_identical_json_passes_and_removes_the_cache(tmp_path):
    # run.sh caches features under <package>/cache; the package must stay as assembled.
    pkg = fake_package(tmp_path / "T07", 'mkdir -p "$ROOT/cache/flow"')
    assert rerun(pkg, tmp_path) == []
    assert not (pkg / "cache").exists()


def test_rerun_problems_existing_cache_is_refused(tmp_path):
    # A warm cache would hide a run that cannot recompute features from the MP4.
    pkg = fake_package(tmp_path / "T07", "")
    (pkg / "cache").mkdir()
    got = rerun(pkg, tmp_path)
    assert len(got) == 1 and "empty cache" in got[0]


def test_rerun_problems_different_json_is_reported(tmp_path):
    pkg = fake_package(tmp_path / "T07", 'echo "{\\"x\\": 1}" > "$2"; exit 0')
    got = rerun(pkg, tmp_path)
    assert len(got) == 1 and "differs" in got[0]


def test_rerun_problems_fallback_logged_is_reported_even_with_identical_json(tmp_path):
    pkg = fake_package(tmp_path / "T07", 'echo "analysis failed, keeping the fallback" >&2')
    got = rerun(pkg, tmp_path)
    assert len(got) == 1 and "fell back" in got[0]


def test_rerun_problems_exit_0_without_json_is_reported(tmp_path):
    pkg = fake_package(tmp_path / "T07", "exit 0")
    got = rerun(pkg, tmp_path)
    assert len(got) == 1 and "wrote no JSON" in got[0]


def test_rerun_problems_nonzero_exit_is_reported(tmp_path):
    pkg = fake_package(tmp_path / "T07", "exit 2")
    got = rerun(pkg, tmp_path)
    assert len(got) == 1 and "exited 2" in got[0]


def test_rerun_problems_timeout_is_reported(tmp_path):
    pkg = fake_package(tmp_path / "T07", "sleep 30")
    got = rerun(pkg, tmp_path, timeout=0.5)
    assert len(got) == 1 and "no result after" in got[0]


def test_rerun_problems_runs_the_package_code_not_the_repo_code(repo, tmp_path, monkeypatch):
    # Regression (self-review of #44): `python -m rewind.cli` puts the working directory
    # first on sys.path, so a check started from the repo's code/ imported the repo's
    # rewind and passed a broken package.
    out = write_outputs(tmp_path / "out", {"v": 60.0}, "T07")
    pkg = tmp_path / "dist" / "T07"
    ms.assemble(repo, pkg, out, {"v": 60.0}, weights=[])
    (pkg / "code" / "rewind" / "cli.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    monkeypatch.chdir(REPO_ROOT / "code")

    got = ms.rerun_problems(pkg, tmp_path / "v.mp4", ms.sys.executable, timeout=120)
    assert len(got) == 1 and "exited 7" in got[0]


@needs_ffmpeg
def test_rerun_problems_package_without_weights_is_caught_as_fallback(repo, tmp_path):
    # The real package with no weights/: run.sh still exits 0 with valid JSON, and the
    # JSON even equals a fallback made by the repo, so only the fallback check stops it.
    video = make_video(tmp_path / "vid_a.mp4", 12)
    out = tmp_path / "out"
    run_video(video, out / "vid_a.json", CFG, team_id=CFG.team.team_id, analyze=no_models)
    pkg = tmp_path / "dist" / "T07"
    ms.assemble(repo, pkg, out, {"vid_a": 12.0}, weights=[])

    got = ms.rerun_problems(pkg, video, ms.sys.executable, timeout=300)
    assert len(got) == 1 and "fell back" in got[0]


def test_shipped_config_team_id_is_our_registered_team_name():
    # Q11, team decision 2026-11-17: the ID is the registered CollegeName_TeamName.
    team_id = load_config(DEFAULT_CONFIG).team.team_id
    assert ms.team_id_problems(team_id) == []
    assert team_id == "VITVellore_Nexus"


def test_main_with_placeholder_team_id_writes_nothing(tmp_path, capsys, monkeypatch):
    # A config still holding the placeholder (as before Q11 was settled): no package.
    placeholder = replace(CFG, team=replace(CFG.team, team_id="TBD_Q11"))
    monkeypatch.setattr(ms, "load_config", lambda path: placeholder)
    ids = read_manifest(REPO_ROOT / "kit" / "manifest.csv")
    out = write_outputs(tmp_path / "out", ids, "TBD_Q11")
    rc = ms.main(["--outputs", str(out), "--dest", str(tmp_path / "dist")])

    assert rc == 1
    assert not (tmp_path / "dist").exists()
    assert "placeholder" in capsys.readouterr().err
