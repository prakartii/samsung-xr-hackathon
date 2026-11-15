"""The Dockerfile ships everything inference reads, and .dockerignore keeps it in."""

from tests.conftest import REPO_ROOT

DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
IGNORED = {
    line.strip()
    for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
}
# run.sh -> rewind.cli: code/, configs/default.yaml, weights/ (models, prior, released
# features); the class list ships inside the package (rewind/resources).
NEEDED = ["run.sh", "requirements.txt", "code", "configs", "weights"]


def copied() -> set[str]:
    out = set()
    for line in DOCKERFILE.splitlines():
        if line.startswith("COPY "):
            out.update(p.rstrip("/") for p in line.split()[1:-1])
    return out


def test_dockerfile_copies_everything_inference_reads():
    assert set(NEEDED) <= copied()


def test_dockerignore_does_not_drop_anything_inference_reads():
    assert IGNORED.isdisjoint(NEEDED)


def test_dockerfile_entrypoint_is_run_sh():
    assert 'ENTRYPOINT ["./run.sh"]' in DOCKERFILE
