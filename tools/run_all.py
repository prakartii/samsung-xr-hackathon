"""Run the pipeline on every manifest video (P1/P7/P9: regenerate outputs/).

    python tools/run_all.py --manifest kit/manifest.csv --videos data/videos --out outputs/

Same `rewind.pipeline.run_video` as run.sh, in one process, so the models load once.
The manifest only lists which videos to run; inference never sees it. Each output is
`<out>/<video_id>.json`, so `video_id` matches the file name (kit R2). A video with
no readable duration is reported and skipped (run.sh exits 2 there too); every other
failure already becomes the validated fallback inside `run_video`.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))


def plan(
    ids: list[str], videos: Path, out: Path, skip_existing: bool
) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    """(jobs as (mp4, json), ids skipped because their JSON exists, ids with no MP4)."""
    jobs, skipped, missing = [], [], []
    for vid in ids:
        mp4, js = videos / f"{vid}.mp4", out / f"{vid}.json"
        if not mp4.exists():
            missing.append(vid)
        elif skip_existing and js.exists():
            skipped.append(vid)
        else:
            jobs.append((mp4, js))
    return jobs, skipped, missing


def main() -> int:
    from rewind.config import load_config
    from rewind.pipeline import run_video
    from rewind.video_io import VideoError

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=REPO / "kit" / "manifest.csv")
    ap.add_argument("--videos", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--out", type=Path, default=REPO / "outputs")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument("--team-id", help="overrides team.team_id (as REWIND_TEAM_ID does for run.sh)")
    ap.add_argument("--skip-existing", action="store_true", help="resume an interrupted run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with args.manifest.open(newline="", encoding="utf-8") as f:
        ids = [r["video_id"] for r in csv.DictReader(f)]
    args.out.mkdir(parents=True, exist_ok=True)
    jobs, skipped, missing = plan(ids, args.videos, args.out, args.skip_existing)
    print(f"{len(jobs)} to run, {len(skipped)} already done, {len(missing)} without an MP4")

    failed, t0 = [], time.perf_counter()
    for i, (mp4, js) in enumerate(jobs, 1):
        t = time.perf_counter()
        try:
            run_video(mp4, js, cfg, team_id=args.team_id or cfg.team.team_id)
        except VideoError as exc:
            failed.append(js.stem)
            print(f"[{i}/{len(jobs)}] {js.stem}: NO OUTPUT ({exc})", flush=True)
            continue
        print(f"[{i}/{len(jobs)}] {js.stem}: {time.perf_counter() - t:.0f} s", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f} s; missing MP4: {missing}; no output: {failed}")
    return 1 if missing or failed else 0


if __name__ == "__main__":
    sys.exit(main())
