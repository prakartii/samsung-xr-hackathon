"""run.sh and the pipeline on tiny synthetic videos (CLAUDE.md §7.2).

Videos are generated with ffmpeg's testsrc in a session fixture (no binaries in git).
Every test checks the JSON written, not internals.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from rewind.config import load_config
from rewind.pipeline import run_video
from rewind.schema import validate
from rewind.types import VideoResult
from rewind.vocab import load_vocab
from tests.conftest import DEFAULT_CONFIG, REPO_ROOT

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def make_video(path: Path, seconds: int) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", f"testsrc=duration={seconds}:size=160x120:rate=10",
            "-pix_fmt", "yuv420p", "-y", str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


@pytest.fixture(autouse=True)
def _no_models(no_model_config, monkeypatch):
    # run.sh and the CLI read REWIND_CONFIG: every test here takes the fallback path.
    monkeypatch.setenv("REWIND_CONFIG", str(no_model_config))


@pytest.fixture(scope="session")
def videos(tmp_path_factory) -> dict[int, Path]:
    d = tmp_path_factory.mktemp("videos")
    return {s: make_video(d / f"clip{s}.mp4", s) for s in (30, 12)}


def run_sh(
    video: Path, out: Path, root: Path = REPO_ROOT, **env_extra: str
) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHON": sys.executable, **env_extra}
    return subprocess.run(
        [str(root / "run.sh"), str(video), str(out)], capture_output=True, text=True, env=env
    )


def strict_errors(path: Path) -> list[str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    errors, _ = validate(doc, load_vocab(), filename_stem=path.stem, strict=True)
    return errors


def test_run_sh_30s_video_writes_strictly_valid_json(videos, tmp_path):
    out = tmp_path / "P01_11.json"

    proc = run_sh(videos[30], out)

    assert proc.returncode == 0, proc.stderr
    assert strict_errors(out) == []
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["video_id"] == "P01_11"  # from the output name, not the input
    assert doc["duration_sec"] == pytest.approx(30.0, abs=0.5)


def test_run_sh_12s_video_writes_one_highlight(videos, tmp_path):
    out = tmp_path / "short.json"

    proc = run_sh(videos[12], out)

    assert proc.returncode == 0, proc.stderr
    assert strict_errors(out) == []
    # T = 12 < 27 s: exactly one 4 s-class window (ARCHITECTURE.md §5.6).
    assert len(json.loads(out.read_text(encoding="utf-8"))["highlights"]) == 1


def test_run_sh_team_id_from_environment(videos, tmp_path):
    out = tmp_path / "v.json"

    proc = run_sh(videos[30], out, REWIND_TEAM_ID="T99")

    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["team_id"] == "T99"


def test_run_sh_wrong_argument_count_prints_usage_and_exits_2():
    proc = subprocess.run([str(REPO_ROOT / "run.sh"), "only-one"], capture_output=True, text=True)

    assert proc.returncode == 2
    assert proc.stderr.startswith("usage:")


def test_run_sh_missing_video_exits_2_without_writing(tmp_path):
    out = tmp_path / "x.json"

    proc = run_sh(tmp_path / "nope.mp4", out)

    assert proc.returncode == 2
    assert not out.exists()


def test_run_sh_works_without_kit_directory(videos, tmp_path):
    # Organizers re-run run.sh in isolation: copy only what the submission ships.
    root = tmp_path / "TEAM"
    shutil.copytree(REPO_ROOT / "code", root / "code", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO_ROOT / "configs", root / "configs")
    shutil.copy2(REPO_ROOT / "run.sh", root / "run.sh")
    out = tmp_path / "out" / "clip.json"

    proc = run_sh(videos[30], out, root=root)

    assert proc.returncode == 0, proc.stderr
    assert strict_errors(out) == []


def test_run_sh_output_does_not_depend_on_input_filename(videos, tmp_path):
    # FPVSum names carry the category (Ski_...): renaming must change nothing but video_id.
    a = shutil.copy2(videos[30], tmp_path / "Ski_-Z5a4UQmZLs.mp4")
    b = shutil.copy2(videos[30], tmp_path / "anything.mp4")
    out_a, out_b = tmp_path / "a" / "same.json", tmp_path / "b" / "same.json"

    run_sh(Path(a), out_a)
    run_sh(Path(b), out_b)

    assert out_a.read_bytes() == out_b.read_bytes()


def test_run_sh_is_deterministic(videos, tmp_path):
    out1, out2 = tmp_path / "1" / "v.json", tmp_path / "2" / "v.json"

    run_sh(videos[30], out1)
    run_sh(videos[30], out2)

    assert out1.read_bytes() == out2.read_bytes()


def test_run_sh_never_opens_manifest_or_ground_truth(videos, tmp_path):
    # An audit hook records every file the inference process opens.
    script = (
        "import sys\n"
        "opened = []\n"
        "sys.addaudithook(lambda ev, a: opened.append(str(a[0])) if ev == 'open' else None)\n"
        "from rewind.cli import main\n"
        f"rc = main([{str(videos[30])!r}, {str(tmp_path / 'v.json')!r}])\n"
        "print('\\n'.join(opened))\n"
        "sys.exit(rc)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "code")},
    )
    assert proc.returncode == 0, proc.stderr
    opened = [p.lower() for p in proc.stdout.splitlines()]
    forbidden = ("manifest", "annotation", "epic_100_", "/kit/", "/data/")
    # Library files inside the Python installation are not inputs: torch ships
    # `torch/jit/annotations.py`, OpenCV `cv2/data/`. Check everything else.
    install = tuple(
        str(Path(d)).lower()
        for d in {sysconfig.get_paths()[k] for k in ("stdlib", "platstdlib", "purelib", "platlib")}
    )
    inputs = [p for p in opened if not str(Path(p)).lower().startswith(install)]

    assert [p for p in inputs if any(f in p for f in forbidden)] == []
    assert any(p.endswith("v.json") or ".v.json." in p for p in opened)  # hook works


# --- in-process: failure paths (the analyze stage is the edge we fake) -------


def test_run_sh_writes_fallback_when_opencv_cannot_load(videos, tmp_path):
    # A bare Linux image without libGL can't import cv2. run.sh must still reach its
    # fallback and write valid JSON (CLAUDE.md §6.7), so nothing on the path from the CLI
    # to the fallback may import cv2 at module level.
    fake = tmp_path / "fakecv2"
    fake.mkdir()
    (fake / "cv2.py").write_text(
        'raise ImportError("libGL.so.1: cannot open shared object file")\n'
    )
    out = tmp_path / "v.json"
    env = {**os.environ, "PYTHONPATH": f"{fake}{os.pathsep}{REPO_ROOT / 'code'}"}

    proc = subprocess.run(
        [sys.executable, "-m", "rewind.cli", str(videos[30]), str(out)],
        capture_output=True,
        text=True,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    assert strict_errors(out) == []


def test_pipeline_hard_kill_during_analysis_leaves_valid_fallback(videos, tmp_path):
    # Regression: MediaPipe is SIGKILLed while loading on some machines (seen on an
    # Intel Core Ultra laptop, kernel 7.2). No except clause runs on SIGKILL, so the
    # fallback must already be on disk before analysis starts.
    out = tmp_path / "v.json"
    script = (
        "import os, signal, sys\n"
        "from pathlib import Path\n"
        "from rewind.config import load_config\n"
        "from rewind.pipeline import run_video\n"
        "def killed(*_):\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n"
        f"run_video(Path(sys.argv[1]), Path(sys.argv[2]), load_config({str(DEFAULT_CONFIG)!r}),"
        " team_id='T07', analyze=killed)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "code")}

    proc = subprocess.run(
        [sys.executable, "-c", script, str(videos[30]), str(out)], env=env, capture_output=True
    )

    assert proc.returncode == -signal.SIGKILL
    assert strict_errors(out) == []


def test_pipeline_stage_exception_writes_valid_fallback(videos, tmp_path, caplog):
    def broken(*_):
        raise RuntimeError("detector exploded")

    out = tmp_path / "v.json"
    run_video(videos[30], out, load_config(DEFAULT_CONFIG), team_id="T07", analyze=broken)

    assert strict_errors(out) == []
    assert json.loads(out.read_text(encoding="utf-8"))["actions"] == []
    assert "detector exploded" in caplog.text


def test_pipeline_invalid_highlights_are_replaced_keeping_valid_label(videos, tmp_path):
    def bad_highlights(_video, probe, video_id, cfg):
        # A 2 s highlight fails R11/R12; the label and the action are fine.
        acts = (((1.0, 3.0), "open", "fridge", 0.9),)
        return VideoResult(
            "T07", video_id, probe.duration_sec, ("cooking", 0.9), acts, ((1.0, 3.0),)
        )

    out = tmp_path / "v.json"
    run_video(videos[30], out, load_config(DEFAULT_CONFIG), team_id="T07", analyze=bad_highlights)

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert strict_errors(out) == []
    assert doc["label"]["name"] == "cooking"  # kept
    assert [a["verb"] for a in doc["actions"]] == ["open"]  # kept
    assert [h["segment"] for h in doc["highlights"]] != [[1.0, 3.0]]  # replaced


def test_pipeline_out_of_vocabulary_action_is_dropped_not_the_whole_result(videos, tmp_path):
    def one_bad_verb(_video, probe, video_id, cfg):
        acts = (((1.0, 3.0), "open", "fridge", 0.9), ((5.0, 7.0), "teleport", "fridge", 0.8))
        return VideoResult(
            "T07", video_id, probe.duration_sec, ("cooking", 0.9), acts, ((10.0, 14.05),)
        )

    out = tmp_path / "v.json"
    run_video(videos[30], out, load_config(DEFAULT_CONFIG), team_id="T07", analyze=one_bad_verb)

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert strict_errors(out) == []
    assert [a["verb"] for a in doc["actions"]] == ["open"]
    assert doc["highlights"] == [{"segment": [10.0, 14.05]}]
    assert doc["label"]["name"] == "cooking"


def test_run_sh_output_file_is_world_readable(videos, tmp_path):
    out = tmp_path / "v.json"

    run_sh(videos[30], out)

    assert out.stat().st_mode & 0o777 == 0o644


def test_pipeline_team_id_override_is_written(videos, tmp_path, no_model_config):
    out = tmp_path / "v.json"

    run_video(videos[30], out, load_config(no_model_config), team_id="T42")

    assert json.loads(out.read_text(encoding="utf-8"))["team_id"] == "T42"


def test_fallback_for_every_manifest_video_passes_directory_validation(tmp_path):
    # Stand-in for the P1 exit check until the videos are downloaded: the fallback at
    # each of the 187 real durations (including the 11.13 s, 15.28 s and 25.16 s videos)
    # passes strict validation as a complete submission directory.
    from rewind.output import dumps, to_document
    from rewind.pipeline import _fallback
    from rewind_eval.validate import DEFAULT_MANIFEST, read_manifest, validate_dir

    manifest = read_manifest(DEFAULT_MANIFEST)
    cfg = load_config(DEFAULT_CONFIG)
    for video_id, duration in manifest.items():
        doc = to_document(_fallback(video_id, duration, cfg))
        (tmp_path / f"{video_id}.json").write_text(dumps(doc), encoding="utf-8")

    errors, _ = validate_dir(tmp_path, manifest, strict=True)

    assert len(manifest) == 187
    assert errors == {}
