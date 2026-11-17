"""The submission Docker image on real videos with real weights (tier: slow, data).

What the organizers do (kit §6): build the image, run ./run.sh on sample videos in an
isolated container. CI only smoke-tests the fallback path (no large weights); this
checks the full models inside the image:

- the JSON passes strict validation at the manifest duration,
- no stage degraded (fallback, salvage, zero-shot detector, uniform prior, missing
  weights),
- two runs in fresh containers give byte-identical JSON (CLAUDE.md §6.6),
- an EPIC video gets ActionFormer actions (released features found by fingerprint).

Needs: docker, an image built with a full weights/ (`docker build -t rewind .`, or set
REWIND_IMAGE), FPVSum and/or EPIC videos in data/videos. Skips with a reason when a
piece is missing. Each run is a fresh container with an empty cache.
"""

import csv
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from rewind.schema import validate
from rewind.vocab import load_vocab
from tests.conftest import REPO_ROOT

pytestmark = [pytest.mark.slow, pytest.mark.data]

IMAGE = os.environ.get("REWIND_IMAGE", "rewind")
VIDEOS = REPO_ROOT / "data" / "videos"
MANIFEST = REPO_ROOT / "kit" / "manifest.csv"
# Log lines (code/rewind/analyze.py, pipeline.py) that mean a stage degraded.
DEGRADED = (
    "keeping the fallback",
    "salvaging",
    "zero-shot",
    "uniform verb-noun prior",
    "uniform prior",
    "checkpoints missing",
    "failed",
)


@pytest.fixture(scope="module")
def image() -> str:
    if shutil.which("docker") is None:
        pytest.skip("docker not on PATH")
    probe = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    if probe.returncode != 0:
        pytest.skip(f"image {IMAGE!r} not built: docker build -t {IMAGE} .")
    return IMAGE


def shortest_present(dataset: str) -> tuple[str, float]:
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    present = sorted(
        (float(r["duration_sec"]), r["video_id"])
        for r in rows
        if r["dataset"] == dataset and (VIDEOS / f"{r['video_id']}.mp4").exists()
    )
    if not present:
        pytest.skip(f"no {dataset} video in {VIDEOS}: tools/download_videos.py")
    duration, vid = present[0]
    return vid, duration


def run_in_image(image: str, video_id: str, out_dir: Path) -> tuple[dict, bytes, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.chmod(0o777)  # the container's user writes here
    name = f"rewind-test-{uuid.uuid4().hex[:12]}"
    try:
        proc = subprocess.run(
            ["docker", "run", "--rm", "--name", name, "--network", "none",
             "-v", f"{VIDEOS}:/in:ro", "-v", f"{out_dir}:/out",
             image, f"/in/{video_id}.mp4", f"/out/{video_id}.json"],
            capture_output=True, text=True, timeout=3600,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        # Killing the docker CLI leaves the container running: stop it too.
        subprocess.run(["docker", "kill", name], capture_output=True)
        raise
    assert proc.returncode == 0, proc.stderr[-2000:]
    raw = (out_dir / f"{video_id}.json").read_bytes()
    return json.loads(raw), raw, proc.stderr


def degraded_lines(stderr: str) -> list[str]:
    return [ln for ln in stderr.splitlines() if any(m in ln for m in DEGRADED)]


@pytest.fixture(scope="module")
def fpvsum_runs(image, tmp_path_factory):
    vid, duration = shortest_present("fpvsum")
    first = run_in_image(image, vid, tmp_path_factory.mktemp("run1"))
    second = run_in_image(image, vid, tmp_path_factory.mktemp("run2"))
    return vid, duration, first, second


def test_docker_fpvsum_video_is_valid_and_uses_every_model(fpvsum_runs):
    vid, duration, (doc, _, stderr), _ = fpvsum_runs

    errors, _ = validate(
        doc, load_vocab(), filename_stem=vid, expected_duration=duration, strict=True
    )
    assert errors == []
    assert degraded_lines(stderr) == []
    # The router is expected to send this video to the FPVSum route (not the uncertain
    # route, which would add zero-shot actions): a model outcome, deterministic for it.
    assert any("route fpvsum" in ln for ln in stderr.splitlines())
    # The FPVSum route: no kitchen actions, a non-cooking label.
    assert doc["actions"] == []
    assert doc["label"]["name"] not in ("cooking", "other")


def test_docker_two_fresh_containers_give_byte_identical_json(fpvsum_runs):
    _, _, (_, first, err1), (_, second, err2) = fpvsum_runs
    # Two identical fallbacks would also be equal: both runs must be the full pipeline.
    assert degraded_lines(err1) == degraded_lines(err2) == []
    assert first == second


def test_docker_epic_video_gets_actionformer_actions(image, tmp_path):
    vid, duration = shortest_present("epic_kitchens_100")
    doc, _, stderr = run_in_image(image, vid, tmp_path / "out")

    errors, _ = validate(
        doc, load_vocab(), filename_stem=vid, expected_duration=duration, strict=True
    )
    assert errors == []
    assert degraded_lines(stderr) == []  # incl. "no SlowFast features ...: zero-shot"
    assert doc["label"]["name"] == "cooking"
    assert len(doc["actions"]) > 0
