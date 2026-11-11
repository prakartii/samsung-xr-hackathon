"""Download the 187 manifest videos (kit/TEAM_README.md §2) into one flat folder.

Offline data tool, not inference: reading kit/manifest.csv here is fine
(CLAUDE.md §6.1 forbids it only inside code/rewind/).

Steps (run all by default, or pick with --only):

  fpvsum  49 FPVSum videos from YouTube with yt-dlp, saved as <root>/videos/<id>.mp4.
          YouTube ids come from FPVSum's official list (crawler/FPVSum_videolist.csv,
          pinned below), NOT from the name: e.g. `BikePolo_5yf797Lh4Oc_1` is YouTube
          `6oOSbkUiJY0`. Progressive MP4 only (`best[ext=mp4]`), so no ffmpeg is
          needed to merge streams; resolution is ample for 160 px flow and 224 px CLIP.
  epic    138 EPIC-KITCHENS-100 val videos (~184 GB) from the official data.bris
          endpoint. URLs, EPIC-55 splits and MD5s are read as data from a clone of
          github.com/epic-kitchens/epic-kitchens-download-scripts (data/*.csv); we
          don't run its script. Same URL scheme and output layout as that script
          (<root>/EPIC-KITCHENS/<P..>/videos/<id>.MP4), plus resume: partial files
          continue with HTTP Range requests, and each file is MD5-verified before
          it gets its final name.
  link    Hard-link every EPIC .MP4 into <root>/videos/<id>.mp4 (same volume, no copy).

Example (Windows, data on D:)::

    git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts \
        D:/rewind-data/epic-kitchens-download-scripts
    python tools/download_videos.py --root D:/rewind-data \
        --epic-scripts D:/rewind-data/epic-kitchens-download-scripts

Re-running is safe: finished files are skipped. Ends with a report of missing ids
and of videos whose duration is off from the manifest by more than 0.5 s.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2

EPIC_55_BASE = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d"
EPIC_100_BASE = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"
REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "kit" / "manifest.csv"
FPVSUM_LIST_URL = (
    "https://raw.githubusercontent.com/azuxmioy/fpvsum/"
    "9af31f9027ef2b3f5f34a39bf17df5b120a9cc25/crawler/FPVSum_videolist.csv"
)
DURATION_TOL = 0.5  # kit §3: duration within ±0.5 s of the manifest


def read_manifest(path: Path) -> dict[str, dict[str, float]]:
    """{dataset: {video_id: duration_sec}}"""
    by_dataset: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_dataset.setdefault(row["dataset"], {})[row["video_id"]] = float(row["duration_sec"])
    return by_dataset


def fpvsum_youtube_ids() -> dict[str, str]:
    """{video_name: youtube_id} from FPVSum's official list at a pinned commit."""
    with urllib.request.urlopen(FPVSUM_LIST_URL) as resp:
        lines = resp.read().decode("utf-8").splitlines()
    return {row["video_name"]: row["youtube_id"] for row in csv.DictReader(lines)}


def download_fpvsum(ids: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    yt = fpvsum_youtube_ids()
    for i, vid in enumerate(ids, 1):
        target = out_dir / f"{vid}.mp4"
        if target.exists():
            print(f"[fpvsum {i}/{len(ids)}] exists: {vid}", flush=True)
            continue
        print(f"[fpvsum {i}/{len(ids)}] downloading {vid}", flush=True)
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", "best[ext=mp4]/best",
            "--no-playlist", "--retries", "10",
            "-o", str(out_dir / f"{vid}.%(ext)s"),
            f"https://www.youtube.com/watch?v={yt[vid]}",
        ]  # fmt: skip
        # YouTube format extraction needs a JS runtime: put the venv's `deno` (pip
        # package) on PATH for yt-dlp without requiring a system-wide install.
        env = {
            **os.environ,
            "PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")]),
        }
        if subprocess.run(cmd, env=env).returncode != 0:
            print(f"[fpvsum] FAILED: {vid}", flush=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def epic_remote(vid: str, splits55: dict[str, str]) -> tuple[str, list[str], str]:
    """(base url, remote path parts, md5 version) for one EPIC video.

    Mirrors epic_downloader.py: EPIC-55 ids (`P01_11`) live under
    videos/<split>/<P01>/ on the EPIC-55 dataset; extension ids (`P01_101`) under
    <P01>/videos/ on the EPIC-100 dataset.
    """
    participant, num = vid.split("_")
    if len(num) == 3:
        return EPIC_100_BASE, [participant, "videos", f"{vid}.MP4"], "100"
    return EPIC_55_BASE, ["videos", splits55[vid], participant, f"{vid}.MP4"], "55"


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_resumable(url: str, part: Path, label: str) -> None:
    """Download `url` into `part`, continuing from its current size (HTTP Range)."""
    done = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # range starts at the end: already complete
            return
        raise
    with resp:
        if done and resp.status != 206:  # server ignored the range: start over
            done = 0
        total = done + int(resp.headers.get("Content-Length", 0))
        next_report = 0.0
        with part.open("ab" if done else "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done / total >= next_report:
                    print(f"  {label}: {done / 2**30:.2f} / {total / 2**30:.2f} GiB", flush=True)
                    next_report += 0.25


def download_epic(ids: list[str], root: Path, scripts: Path, retries: int = 5) -> None:
    data = scripts / "data"
    splits55 = {r["video_id"]: r["split"] for r in _read_csv(data / "epic_55_splits.csv")}
    md5s = {(r["version"], r["file_remote_path"]): r["md5"] for r in _read_csv(data / "md5.csv")}
    for i, vid in enumerate(ids, 1):
        label = f"[epic {i}/{len(ids)}] {vid}"
        base, parts, version = epic_remote(vid, splits55)
        target = root / "EPIC-KITCHENS" / vid.split("_")[0] / "videos" / f"{vid}.MP4"
        ok_marker = target.with_name(target.name + ".md5ok")
        part = target.with_name(target.name + ".part")
        expected = md5s.get((version, "/".join(parts)))
        if target.exists() and ok_marker.exists():
            print(f"{label}: exists (verified)", flush=True)
            continue
        if target.exists():
            # Unverified file, e.g. a partial left by the official script, which
            # writes straight to the final name. Keep it if it checks out, else
            # resume from it.
            if expected and md5_file(target) == expected:
                ok_marker.touch()
                print(f"{label}: exists (md5 ok)", flush=True)
                continue
            print(f"{label}: existing file fails MD5, resuming it", flush=True)
            os.replace(target, part)
        target.parent.mkdir(parents=True, exist_ok=True)
        url = "/".join([base, *parts])
        print(f"{label}: downloading {url}", flush=True)
        for attempt in range(1, retries + 1):
            try:
                fetch_resumable(url, part, label)
                break
            except (urllib.error.URLError, OSError) as e:
                print(f"{label}: attempt {attempt} failed ({e}); resuming", flush=True)
        else:
            print(f"{label}: FAILED after {retries} attempts", flush=True)
            continue
        if expected and md5_file(part) != expected:
            print(f"{label}: MD5 MISMATCH, deleting partial file", flush=True)
            part.unlink()
            continue
        os.replace(part, target)
        if expected:
            ok_marker.touch()
        print(f"{label}: done{' (md5 ok)' if expected else ' (no md5 listed)'}", flush=True)


def link_epic(ids: list[str], root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for vid in ids:
        src = root / "EPIC-KITCHENS" / vid.split("_")[0] / "videos" / f"{vid}.MP4"
        dst = out_dir / f"{vid}.mp4"
        if src.exists() and not dst.exists():
            os.link(src, dst)


def decoded_duration(path: Path) -> float | None:
    """Duration from the container via OpenCV (frame count / fps); None if unreadable."""
    cap = cv2.VideoCapture(str(path))
    try:
        fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return n / fps if fps > 0 and n > 0 else None
    finally:
        cap.release()


def report(by_dataset: dict[str, dict[str, float]], out_dir: Path) -> None:
    expected = {v: d for ids in by_dataset.values() for v, d in ids.items()}
    missing = [v for v in expected if not (out_dir / f"{v}.mp4").exists()]
    print(f"\n{len(expected) - len(missing)}/{len(expected)} videos present in {out_dir}")
    for v in missing:
        print(f"  missing: {v}")
    for v, want in expected.items():
        if v in missing:
            continue
        got = decoded_duration(out_dir / f"{v}.mp4")
        if got is None or abs(got - want) > DURATION_TOL:
            print(f"  duration mismatch: {v} manifest {want:.2f} s, file {got} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="data root, e.g. D:/rewind-data")
    ap.add_argument("--epic-scripts", type=Path, help="clone of epic-kitchens-download-scripts")
    ap.add_argument("--only", choices=["fpvsum", "epic", "link"], action="append")
    args = ap.parse_args()

    steps = args.only or ["fpvsum", "epic", "link"]
    by_dataset = read_manifest(MANIFEST)
    out_dir = args.root / "videos"

    if "fpvsum" in steps:
        download_fpvsum(list(by_dataset["fpvsum"]), out_dir)
    if "epic" in steps:
        if args.epic_scripts is None:
            ap.error("--epic-scripts is required for the epic step")
        download_epic(list(by_dataset["epic_kitchens_100"]), args.root, args.epic_scripts)
    if "link" in steps:
        link_epic(list(by_dataset["epic_kitchens_100"]), args.root, out_dir)
    report(by_dataset, out_dir)


if __name__ == "__main__":
    main()
