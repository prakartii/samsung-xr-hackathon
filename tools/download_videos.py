"""Download the 187 manifest videos (kit/TEAM_README.md §2) into one flat folder.

Offline data tool, not inference: reading kit/manifest.csv here is fine
(CLAUDE.md §6.1 forbids it only inside code/rewind/).

Steps (run all by default, or pick with --only):

  fpvsum  49 FPVSum videos from YouTube with yt-dlp, saved as <root>/videos/<id>.mp4.
          YouTube ids come from FPVSum's official list (crawler/FPVSum_videolist.csv,
          pinned below), NOT from the name: e.g. `BikePolo_5yf797Lh4Oc_1` is YouTube
          `6oOSbkUiJY0`. Progressive MP4 only (`best[ext=mp4]`), so no ffmpeg is
          needed to merge streams; resolution is ample for 160 px flow and 224 px CLIP.
  epic    138 EPIC-KITCHENS-100 val videos (~184 GB) with the official downloader
          (github.com/epic-kitchens/epic-kitchens-download-scripts), which verifies
          MD5s and skips files already downloaded. It writes
          <root>/EPIC-KITCHENS/<P..>/videos/<id>.MP4.
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
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2

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
        if subprocess.run(cmd).returncode != 0:
            print(f"[fpvsum] FAILED: {vid}", flush=True)


def download_epic(ids: list[str], root: Path, scripts: Path) -> None:
    # The official script resolves its data/*.csv paths relative to the cwd.
    cmd = [
        sys.executable, "-u", "epic_downloader.py", "--videos",
        "--specific-videos", ",".join(ids),
        "--output-path", str(root),
    ]  # fmt: skip
    subprocess.run(cmd, cwd=scripts, check=True)


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
