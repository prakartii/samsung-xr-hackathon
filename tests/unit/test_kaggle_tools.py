"""Pure helpers of the Kaggle caching tools (the Kaggle-side code itself can't run here)."""

import importlib.util
import queue

import numpy as np
import pytest

from tests.conftest import REPO_ROOT


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_and_report_reports_a_crashed_download_as_failed():
    # IncompleteRead (dropped connection) isn't an OSError; the shard must still hear
    # back, or ready.get() blocks until Kaggle's 12 h limit.
    import http.client

    k = _load("kaggle_cache_epic")
    ready: queue.Queue = queue.Queue()

    def crash(_vid):
        raise http.client.IncompleteRead(b"partial")

    k.fetch_and_report("P01_11", crash, ready)

    assert ready.get_nowait() == ("P01_11", False)


def test_fetch_and_report_passes_success_through():
    k = _load("kaggle_cache_epic")
    ready: queue.Queue = queue.Queue()

    k.fetch_and_report("P01_11", lambda _vid: True, ready)

    assert ready.get_nowait() == ("P01_11", True)


def test_pinned_reads_versions_from_requirements(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==2.4.6\nav==18.0.0   # comment\nopen_clip_torch==4.2.2\n")
    k = _load("kaggle_cache_epic")

    assert k.pinned(req, ["av", "open_clip_torch"]) == ["av==18.0.0", "open_clip_torch==4.2.2"]
    with pytest.raises(ValueError, match="mediapipe"):
        k.pinned(req, ["mediapipe"])


def test_kaggle_job_installs_the_repo_pins():
    # The real requirements.txt must pin everything the Kaggle job installs.
    k = _load("kaggle_cache_epic")

    pins = k.pinned(REPO_ROOT / "requirements.txt", ["av", "open_clip_torch", "mediapipe"])

    assert all("==" in p for p in pins)


def _entry(path, version):
    np.savez(path, x=np.zeros(1), __cache_version__=np.asarray(version))


def test_collect_replaces_a_stale_local_entry(tmp_path):
    launch = _load("kaggle_launch")
    src, local = tmp_path / "src.npz", tmp_path / "local.npz"
    _entry(src, "flow-v2")

    assert launch.needs_copy(src, local)  # missing locally
    _entry(local, "flow-v1")
    assert launch.needs_copy(src, local)  # stale version locally
    _entry(local, "flow-v2")
    assert not launch.needs_copy(src, local)  # same version: keep


def test_only_ids_accepts_known_ids_and_strips_spaces():
    launch = _load("kaggle_launch")
    assert launch.parse_only_ids("P08_09, P29_05", {"P08_09", "P29_05", "P01_11"}) == [
        "P08_09",
        "P29_05",
    ]


def test_only_ids_rejects_an_unknown_id():
    # A typo (P8_09) must stop the launch, not silently retry nothing.
    launch = _load("kaggle_launch")
    with pytest.raises(SystemExit, match="P8_09"):
        launch.parse_only_ids("P8_09,P29_05", {"P08_09", "P29_05"})


def test_only_ids_are_all_epic_ids_in_the_kit_manifest():
    launch = _load("kaggle_launch")
    ids = launch.epic_ids()
    assert len(ids) == 138  # EPIC-KITCHENS-100 val videos in kit/manifest.csv
    assert "P08_09" in ids and not any(v.startswith("Ski_") for v in ids)
