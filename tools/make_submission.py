"""Assemble the submission folder TEAM_ID/ (P9; kit §6, ARCHITECTURE.md §13).

    python tools/make_submission.py --outputs outputs/ --dest dist/
    python tools/make_submission.py --outputs outputs/ --dest dist/ \
        --check-video data/videos/<video_id>.mp4 --zip

Refuses to write anything if the package would not be submittable:
  - `team.team_id` in configs/default.yaml is still a placeholder (`TBD…`), or an
    output's `team_id` differs from it (re-running run.sh would then not reproduce it);
  - outputs/ is missing a manifest video, or any output fails strict validation;
  - a weight file named in the config is missing (run.sh would silently fall back);
  - a shipped text file mentions a manifest video ID (kit §6: automated scans for
    hardcoded IDs disqualify).

The package holds run.sh, code/ (rewind, rewind_eval, third_party), configs/default.yaml,
weights/ (only the files the config names), requirements.txt, README.md, the Dockerfile
and outputs/ (one JSON per manifest video). No tests, tools, kit, data or cache.

--check-video runs the *packaged* run.sh on a video, as the organizers will, and
fails unless the JSON is byte-identical to the packaged output for that video.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from rewind.config import Config, load_config  # noqa: E402
from rewind_eval.validate import read_manifest, validate_dir  # noqa: E402

# Repo paths copied as they are (directories recursively, minus bytecode).
SHIPPED = (
    "run.sh",
    "requirements.txt",
    "README.md",
    "Dockerfile",
    ".dockerignore",
    "code/rewind",
    "code/rewind_eval",
    "code/third_party",
    "configs/default.yaml",
)
_SCAN_MAX_BYTES = 5_000_000  # larger files are weights/features, not text


def team_id_problems(team_id: str) -> list[str]:
    if not team_id.strip() or team_id.upper().startswith("TBD"):
        return [f"team.team_id is a placeholder ({team_id!r}): set the assigned ID (Q11)"]
    return []


def weight_paths(cfg: Config) -> list[str]:
    """Every config value that names a path under weights/, sorted."""
    found: set[str] = set()

    def walk(node: object) -> None:
        if dataclasses.is_dataclass(node):
            for f in dataclasses.fields(node):
                walk(getattr(node, f.name))
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)
        elif isinstance(node, str) and node.startswith("weights/"):
            found.add(node)

    walk(cfg)
    return sorted(found)


def missing_weights(root: Path, paths: Iterable[str]) -> list[str]:
    """Paths absent under root (an empty directory counts as absent)."""
    out = []
    for p in paths:
        target = root / p
        if not target.exists() or (target.is_dir() and not any(target.iterdir())):
            out.append(f"missing weight: {p} (run tools/fetch_weights.sh)")
    return out


def output_problems(outputs: Path, manifest: dict[str, float], team_id: str) -> list[str]:
    """Strict validation of the whole directory, plus team_id == the config's."""
    errors, _ = validate_dir(outputs, manifest, strict=True)
    problems = [f"{name}: {e}" for name in sorted(errors) for e in errors[name]]
    for vid in sorted(manifest):
        path = outputs / f"{vid}.json"
        if not path.exists():
            continue  # already reported by validate_dir
        try:
            got = json.loads(path.read_text(encoding="utf-8")).get("team_id")
        except (OSError, ValueError, AttributeError):
            continue  # already reported by validate_dir
        if got != team_id:
            problems.append(f"{path.name}: team_id {got!r} != config team.team_id {team_id!r}")
    return problems


def shipped_files(repo: Path) -> list[Path]:
    """Repo-relative paths of every file SHIPPED expands to."""
    out = []
    for rel in SHIPPED:
        src = repo / rel
        if src.is_file():
            out.append(Path(rel))
        elif src.is_dir():
            for path in sorted(src.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    out.append(path.relative_to(repo))
        else:
            raise FileNotFoundError(f"{rel} is in SHIPPED but not in the repo")
    return out


def id_mentions(root: Path, files: Iterable[Path], ids: Iterable[str]) -> list[str]:
    """`file:line: id` for every manifest video ID in a shipped text file.

    An ID only counts as a whole token: `P01_11` does not match inside `P01_110`.
    """
    alternatives = "|".join(re.escape(i) for i in sorted(ids, key=len, reverse=True))
    if not alternatives:
        return []
    pattern = re.compile(rf"(?<![\w-])(?:{alternatives})(?![\w-])")
    hits = []
    for rel in files:
        path = root / rel
        if path.stat().st_size > _SCAN_MAX_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary
        for n, line in enumerate(text.splitlines(), 1):
            for m in pattern.finditer(line):
                hits.append(f"{rel.as_posix()}:{n}: video ID {m.group(0)}")
    return hits


def weight_files(repo: Path, paths: Iterable[str]) -> list[Path]:
    out = []
    for p in paths:
        src = repo / p
        if src.is_dir():
            out += [f.relative_to(repo) for f in sorted(src.rglob("*")) if f.is_file()]
        elif src.is_file():
            out.append(Path(p))
    return out


def _place(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(src, dst)  # weights are large and read-only: no second copy
            return
        except OSError:
            pass  # other filesystem
    shutil.copy2(src, dst)


def assemble(
    repo: Path, pkg: Path, outputs: Path, ids: Iterable[str], weights: Iterable[Path]
) -> None:
    """Write the package to pkg (must not exist)."""
    pkg.mkdir(parents=True)
    for rel in shipped_files(repo):
        _place(repo / rel, pkg / rel, link=False)
    for rel in weights:
        _place(repo / rel, pkg / rel, link=True)
    for vid in sorted(ids):
        _place(outputs / f"{vid}.json", pkg / "outputs" / f"{vid}.json", link=False)
    (pkg / "run.sh").chmod(0o755)


def rerun_problems(pkg: Path, video: Path, python: str, timeout: float) -> list[str]:
    """Run pkg/run.sh on video; problems unless its JSON equals pkg/outputs/<stem>.json.

    Runs from an empty feature cache, as on the organizers' machine, and removes the
    cache it creates inside pkg so the package stays as assembled.
    """
    expected = pkg / "outputs" / f"{video.stem}.json"
    if not expected.exists():
        return [f"--check-video {video.name}: no packaged output {expected.name}"]
    cache = pkg / "cache"
    if cache.exists():
        return [f"--check-video: {cache} exists; the re-run must start from an empty cache"]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / expected.name
        env = {k: v for k, v in os.environ.items() if not k.startswith("REWIND_")}
        env.update(PYTHON=python, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
        try:
            proc = subprocess.run(
                [str(pkg / "run.sh"), str(video), str(out)],
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
                check=False,
            )
        finally:
            shutil.rmtree(cache, ignore_errors=True)
        if proc.returncode != 0:
            tail = proc.stderr[-500:]
            return [f"--check-video: packaged run.sh exited {proc.returncode}: {tail}"]
        if out.read_bytes() != expected.read_bytes():
            return [f"--check-video: packaged run.sh output differs from outputs/{expected.name}"]
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--outputs", type=Path, required=True, help="directory of <video_id>.json")
    ap.add_argument("--dest", type=Path, required=True, help="the package goes to <dest>/<TEAM_ID>")
    ap.add_argument("--manifest", type=Path, default=REPO / "kit" / "manifest.csv")
    ap.add_argument(
        "--check-video",
        type=Path,
        action="append",
        default=[],
        help="re-run the packaged run.sh on this video and compare (repeatable)",
    )
    ap.add_argument("--python", default=sys.executable, help="interpreter for --check-video")
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds per --check-video run")
    ap.add_argument("--zip", action="store_true", help="also write <dest>/<TEAM_ID>.zip")
    args = ap.parse_args(argv)

    cfg = load_config(REPO / "configs" / "default.yaml")  # the shipped config
    team_id = cfg.team.team_id
    manifest = read_manifest(args.manifest)
    weights = weight_paths(cfg)

    problems = team_id_problems(team_id)
    problems += missing_weights(REPO, weights)
    problems += output_problems(args.outputs, manifest, team_id)
    problems += id_mentions(REPO, shipped_files(REPO), manifest)
    pkg = args.dest / team_id
    if pkg.exists():
        problems.append(f"{pkg} already exists: remove it first")
    if problems:
        for p in problems:
            print(f"ERROR {p}", file=sys.stderr)
        print(f"not assembled: {len(problems)} problem(s)", file=sys.stderr)
        return 1

    assemble(REPO, pkg, args.outputs, manifest, weight_files(REPO, weights))
    print(f"assembled {pkg} ({len(manifest)} outputs, {len(weights)} weight paths)")

    for video in args.check_video:
        errs = rerun_problems(pkg, video, args.python, args.timeout)
        for e in errs:
            print(f"ERROR {e}", file=sys.stderr)
        if errs:
            return 1
        print(f"re-run ok: {video.name} byte-identical to the packaged output")

    if args.zip:
        archive = shutil.make_archive(str(args.dest / team_id), "zip", args.dest, team_id)
        print(f"wrote {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
