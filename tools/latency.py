"""Time the full pipeline per video on this machine's CPU (P7; deck latency number).

    python tools/latency.py --videos data/videos --ids P01_11,Bike_lhTFVa0h9Rk
    python tools/latency.py --videos data/videos --sample 3      # 3 EPIC + 3 FPVSum

Each video runs through `python -m rewind.cli` (what run.sh executes) in a fresh
process with an EMPTY feature cache in a temp dir, so the time includes model
loading, decoding and every feature. The shipped ActionFormer features in
`weights/released_features` are still used, as on a clean submission machine.

Reports seconds of wall time per minute of video, pooled per route (total seconds /
total video minutes) and the slowest video, and appends the runs to
results/latency.csv. The manifest is read only to pick and group videos; inference
never sees it.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Run:
    video_id: str
    route: str  # manifest dataset group: "epic" or "fpvsum"
    duration_sec: float
    seconds: float


def summarize(runs: list[Run]) -> dict[str, dict[str, float]]:
    """Per route and overall: pooled seconds per video minute, slowest video, count."""
    groups: dict[str, list[Run]] = {"all": list(runs)}
    for r in runs:
        groups.setdefault(r.route, []).append(r)
    out = {}
    for name, rs in groups.items():
        minutes = sum(r.duration_sec for r in rs) / 60.0
        out[name] = {
            "videos": len(rs),
            "sec_per_video_min": sum(r.seconds for r in rs) / minutes,
            "worst_sec_per_video_min": max(r.seconds / (r.duration_sec / 60.0) for r in rs),
        }
    return out


def manifest() -> dict[str, tuple[str, float]]:
    with (REPO / "kit" / "manifest.csv").open(newline="", encoding="utf-8") as f:
        return {
            r["video_id"]: (
                "fpvsum" if r["dataset"] == "fpvsum" else "epic",
                float(r["duration_sec"]),
            )
            for r in csv.DictReader(f)
        }


def time_one(video: Path, cfg_path: Path, out_dir: Path) -> float:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "rewind.cli", str(video), str(out_dir / f"{video.stem}.json"),
         "--config", str(cfg_path)],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "code")},
    )  # fmt: skip
    seconds = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"{video.name}: exit {proc.returncode}\n{proc.stderr[-2000:]}")
    return seconds


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--videos", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--ids", help="comma list of video ids")
    ap.add_argument("--sample", type=int, default=0, help="first N videos of each dataset")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "latency.csv")
    args = ap.parse_args()

    meta = manifest()
    if args.ids:
        ids = [v.strip() for v in args.ids.split(",") if v.strip()]
    else:
        groups = ("epic", "fpvsum")
        ids = [v for g in groups for v in sorted(v for v in meta if meta[v][0] == g)[: args.sample]]
    if not ids:
        ap.error("give --ids or --sample")

    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        cfg["features"]["cache_dir"] = str(Path(tmp) / "cache")  # empty cache, every run
        cfg_path = Path(tmp) / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        for vid in ids:
            route, duration = meta[vid]
            shutil.rmtree(Path(tmp) / "cache", ignore_errors=True)  # cold start per video
            seconds = time_one(args.videos / f"{vid}.mp4", cfg_path, Path(tmp))
            runs.append(Run(vid, route, duration, seconds))
            print(f"{vid:28s} {route:6s} {duration:7.1f} s video  {seconds:7.1f} s  "
                  f"{seconds / (duration / 60):6.1f} s/min", flush=True)  # fmt: skip

    for name, s in summarize(runs).items():
        print(f"{name:6s} {s['videos']:3d} videos  {s['sec_per_video_min']:6.1f} s per video min"
              f"  (slowest {s['worst_sec_per_video_min']:.1f})")  # fmt: skip
    new = not args.out.exists()
    with args.out.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["video_id", "route", "duration_sec", "seconds", "machine"])
        machine = f"{sys.platform} cpu"
        for r in runs:
            w.writerow([r.video_id, r.route, f"{r.duration_sec:.2f}", f"{r.seconds:.1f}", machine])


if __name__ == "__main__":
    main()
