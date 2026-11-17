"""tools/fetch_weights.sh against a fake local release (file:// URLs, no network).

The script reads its pinned hashes from WEIGHTS_SHA256SUMS, so the test can serve
small stand-in files and check behavior: every weight the config names ends up in
weights/, a corrupt download is refused, and a re-run downloads nothing.
"""

import hashlib
import importlib.util
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from rewind.config import load_config
from tests.conftest import DEFAULT_CONFIG, REPO_ROOT

pytestmark = pytest.mark.skipif(
    shutil.which("curl") is None or shutil.which("sha256sum") is None,
    reason="needs curl and sha256sum",
)

_spec = importlib.util.spec_from_file_location(
    "make_submission", REPO_ROOT / "tools" / "make_submission.py"
)
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)

CFG = load_config(DEFAULT_CONFIG)
FILES = [
    "hand_landmarker.task",
    "clip_vit_b16_openai.safetensors",
    "actionformer_epic_verb.pth.tar",
    "actionformer_epic_noun.pth.tar",
    "vn_prior.npy",
]


def asset_name(weight_path: str) -> str:
    # weights/released_features is a directory, released as one tar.
    name = Path(weight_path).name
    return name if Path(name).suffix else f"{name}.tar"


def make_release(root: Path) -> tuple[Path, Path]:
    """A release dir holding a stand-in for every asset, and its hash list."""
    release = root / "release"
    release.mkdir()
    for name in FILES:
        (release / name).write_bytes(f"stand-in for {name}\n".encode())
    feats = root / "src" / "released_features" / "slowfast"
    feats.mkdir(parents=True)
    (feats / "abc.npy").write_bytes(b"features")
    with tarfile.open(release / "released_features.tar", "w") as tar:
        tar.add(root / "src" / "released_features", arcname="released_features")
    sums = root / "weights.sha256"
    sums.write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
            for p in sorted(release.iterdir())
        )
    )
    return release, sums


def run_fetch(repo: Path, release: Path, sums: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(repo / "tools" / "fetch_weights.sh")],
        env={
            "PATH": "/usr/bin:/bin",
            "WEIGHTS_RELEASE_URL": release.as_uri(),
            "WEIGHTS_SHA256SUMS": str(sums),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "repo" / "tools").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "tools" / "fetch_weights.sh", tmp_path / "repo" / "tools")
    return tmp_path / "repo"


def test_fetch_weights_from_release_fills_every_configured_weight(repo, tmp_path):
    release, sums = make_release(tmp_path)

    proc = run_fetch(repo, release, sums)

    assert proc.returncode == 0, proc.stderr
    # What run.sh needs: every weights/ path the config names (6), none missing.
    assert ms.missing_weights(repo, ms.weight_paths(CFG)) == []
    assert (repo / "weights" / "released_features" / "slowfast" / "abc.npy").read_bytes() == (
        b"features"
    )
    assert sorted(p.name for p in (repo / "weights").iterdir()) == sorted(
        [*FILES, "released_features"]
    )  # no .tmp, tar or unpack leftovers


def test_fetch_weights_corrupt_download_is_refused(repo, tmp_path):
    release, sums = make_release(tmp_path)
    (release / "clip_vit_b16_openai.safetensors").write_bytes(b"tampered")

    proc = run_fetch(repo, release, sums)

    assert proc.returncode != 0
    assert "SHA-256 mismatch for clip_vit_b16_openai.safetensors" in proc.stderr
    assert not (repo / "weights" / "clip_vit_b16_openai.safetensors").exists()
    assert not (repo / "weights" / "clip_vit_b16_openai.safetensors.tmp").exists()


def test_fetch_weights_rerun_downloads_nothing(repo, tmp_path):
    release, sums = make_release(tmp_path)
    assert run_fetch(repo, release, sums).returncode == 0

    proc = run_fetch(repo, release, sums)

    assert proc.returncode == 0, proc.stderr
    assert "fetching" not in proc.stdout


def test_pinned_hash_list_covers_exactly_the_configured_weights():
    pinned = (REPO_ROOT / "tools" / "weights.sha256").read_text().split("\n")
    names = sorted(line.split()[1] for line in pinned if line.strip())
    assert names == sorted(asset_name(p) for p in ms.weight_paths(CFG))
