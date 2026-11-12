"""Launch the EPIC feature-cache job on Kaggle, check on it, and collect the results.

    python tools/kaggle_launch.py --shards 5 --epic-scripts <epic-kitchens-download-scripts clone>
    python tools/kaggle_launch.py --status --shards 5
    python tools/kaggle_launch.py --collect --shards 5      # merge outputs into ./cache
    python tools/kaggle_launch.py --shards 5 --epic-scripts <clone> --only-ids P08_09,P29_05
                                                            # retry only these videos

Needs the Kaggle CLI and a token in ~/.kaggle (never committed, never printed). Uses
only free CPU notebooks with internet; nothing is billed. Everything is PRIVATE:
- dataset `<user>/rewind-code`: a tar.gz of code/, tools/, configs/, kit/ and the
  official downloader's data CSVs (repo is private, so notebooks can't clone it);
- kernels `<user>/rewind-epic-cache-<i>`: tools/kaggle_cache_epic.py with SHARD = i.

After collecting, expire the token (Kaggle -> Settings -> API).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = str(Path(sys.executable).parent / "kaggle")
PACK = ["code", "tools", "configs", "kit", "requirements.txt"]  # the job installs its pins


def run(*args: str, check: bool = True) -> str:
    out = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    if check and out.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{out.stdout}\n{out.stderr}")
    return out.stdout + out.stderr


def username() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api.get_config_value("username")


def push_code(user: str, epic_scripts: Path, work: Path) -> None:
    ds = work / "dataset"
    ds.mkdir()
    with tarfile.open(ds / "rewind-code.tar.gz", "w:gz") as tf:
        for name in PACK:
            tf.add(
                REPO / name,
                arcname=name,
                filter=lambda ti: None if "__pycache__" in ti.name else ti,
            )
        for csv in ("epic_55_splits.csv", "epic_100_splits.csv", "md5.csv"):
            tf.add(epic_scripts / "data" / csv, arcname=f"epic_download_data/{csv}")
    meta = {"title": "rewind-code", "id": f"{user}/rewind-code", "licenses": [{"name": "other"}]}
    (ds / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    exists = "rewind-code" in run("datasets", "list", "--mine", check=False)
    if exists:
        print(run("datasets", "version", "-p", str(ds), "-m", "update code"))
    else:
        print(run("datasets", "create", "-p", str(ds)))  # private by default
    for _ in range(60):  # kernels can only attach a dataset once Kaggle has processed it
        status = run("datasets", "status", f"{user}/rewind-code", check=False).strip()
        print("dataset status:", status)
        if "ready" in status.lower():
            return
        time.sleep(10)
    raise SystemExit("dataset not ready after 10 minutes")


def parse_only_ids(text: str, known: set[str]) -> list[str]:
    """Split a comma list of EPIC ids; any id not in the kit manifest is an error
    (a typo would otherwise be dropped silently and the retry would do nothing)."""
    ids = [v.strip() for v in text.split(",") if v.strip()]
    unknown = sorted(set(ids) - known)
    if not ids or unknown:
        raise SystemExit(f"--only-ids: not EPIC ids in kit/manifest.csv: {unknown or text!r}")
    return ids


def epic_ids() -> set[str]:
    with (REPO / "kit" / "manifest.csv").open(newline="", encoding="utf-8") as f:
        return {r["video_id"] for r in csv.DictReader(f) if r["dataset"] == "epic_kitchens_100"}


def tune_ids() -> list[str]:
    """The `tune` split's EPIC train videos (configs/splits.yaml); never manifest videos."""
    import yaml

    data = yaml.safe_load((REPO / "configs" / "splits.yaml").read_text(encoding="utf-8"))
    return list(data["tune"]["epic"])


def push_kernels(
    user: str, shards: int, work: Path, only_ids: list[str] | None = None, run_tag: str = ""
) -> None:
    body = (REPO / "tools" / "kaggle_cache_epic.py").read_text(encoding="utf-8")
    for i in range(shards):
        kdir = work / f"kernel_{i}"
        kdir.mkdir()
        header = (
            f"SHARD = {i}\nN_SHARDS = {shards}\nONLY_IDS = {only_ids!r}\nRUN_TAG = {run_tag!r}\n"
        )
        (kdir / "run.py").write_text(header + body, encoding="utf-8")
        meta = {
            "id": f"{user}/rewind-epic-cache-{i}",
            "title": f"rewind-epic-cache-{i}",
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": True,
            "dataset_sources": [f"{user}/rewind-code"],
        }
        (kdir / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        print(run("kernels", "push", "-p", str(kdir)))


def _cache_version(path: Path) -> str | None:
    """The extractor version stored inside a cache entry (None if unreadable)."""
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as npz:
            return str(npz["__cache_version__"]) if "__cache_version__" in npz.files else None
    except (OSError, ValueError):
        return None


def needs_copy(src: Path, target: Path) -> bool:
    """Copy a collected entry unless the local one was built by the same extractor
    version; a stale local entry (older version) must not block the new one."""
    return not target.exists() or _cache_version(target) != _cache_version(src)


def collect(user: str, shards: int, work: Path) -> None:
    dest = REPO / "cache"
    for i in range(shards):
        out = work / f"out_{i}"
        out.mkdir()
        print(
            run("kernels", "output", f"{user}/rewind-epic-cache-{i}", "-p", str(out), check=False)
        )
        for npz in (out / "cache").rglob("*.npz") if (out / "cache").exists() else []:
            target = dest / npz.relative_to(out / "cache")
            target.parent.mkdir(parents=True, exist_ok=True)
            if needs_copy(npz, target):
                shutil.copy2(npz, target)
        logs = dest / "kaggle_logs"
        logs.mkdir(parents=True, exist_ok=True)
        for log in out.glob("shard_*.jsonl"):
            shutil.copy2(log, logs / log.name)
    print(f"merged into {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument("--epic-scripts", type=Path)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--kernels-only", action="store_true", help="re-push kernels, keep dataset")
    ap.add_argument("--only-ids", help="comma list of EPIC ids to (re)process, e.g. failed ones")
    ap.add_argument("--split", choices=["tune"], help="process a split's videos (tune: EPIC train)")
    args = ap.parse_args()
    user = username()
    if args.status:
        for i in range(args.shards):
            print(i, run("kernels", "status", f"{user}/rewind-epic-cache-{i}", check=False).strip())
        return
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        if args.collect:
            collect(user, args.shards, work)
            return
        if not args.kernels_only:
            if args.epic_scripts is None:
                ap.error("--epic-scripts is required to package the downloader CSVs")
            push_code(user, args.epic_scripts, work)
        if args.split == "tune":
            only, tag = tune_ids(), "_tune"
        elif args.only_ids:
            only, tag = parse_only_ids(args.only_ids, epic_ids() | set(tune_ids())), "_retry"
        else:
            only, tag = None, ""
        push_kernels(user, args.shards, work, only, tag)


if __name__ == "__main__":
    main()
