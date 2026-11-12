"""Check FPVSum GT timing against our downloaded videos (P6).

    python tools/check_fpvsum_gt.py

`rewind_eval.gt_fpvsum` spreads each video's N annotated frames evenly over the
manifest duration. That is right only if the annotation covers the same cut as our
YouTube download. If it does, N / duration is a standard frame rate (24, 25, 30, 50,
60 or their NTSC variants); a rate like 21.3 fps means the annotated cut has a
different length, and every GT highlight frame lands at the wrong time.

Prints, per FPVSum video: GT frames, manifest duration, implied GT fps, our file's
fps, and SUSPECT where the implied rate is more than `--tol` from every standard
rate. Offline evaluation tool (reads the eval ground truth).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

STANDARD_FPS = (
    24000 / 1001, 24.0, 25.0, 30000 / 1001, 30.0, 48.0, 50.0, 60000 / 1001, 60.0,
)  # fmt: skip


def gt_rate_suspect(n_gt: int, duration: float, tol: float) -> bool:
    """True if n_gt / duration is more than `tol` (relative) from every standard rate."""
    implied = n_gt / duration
    return min(abs(implied - f) / f for f in STANDARD_FPS) > tol


def main() -> None:
    import numpy as np
    import scipy.io

    from rewind.video_io import probe_video
    from rewind_eval.gt_fpvsum import name_key

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--videos", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--tol", type=float, default=0.02, help="relative distance to a standard fps")
    args = ap.parse_args()

    with (REPO / "kit" / "manifest.csv").open(newline="", encoding="utf-8") as f:
        dur = {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader(f)
               if r["dataset"] == "fpvsum"}  # fmt: skip
    gt_mat = REPO / "data" / "annotations" / "fpvsum" / "gt_data.mat"
    records = scipy.io.loadmat(gt_mat, squeeze_me=True, struct_as_record=False)["gt_data"]
    by_key = {name_key(str(r.Name)): r for r in np.atleast_1d(records)}

    suspects = []
    print(f"{'video':30s} {'gt_frames':>9s} {'dur_s':>7s} {'gt_fps':>7s} {'file_fps':>8s}")
    for vid, T in sorted(dur.items()):
        n_gt = len(np.asarray(by_key[name_key(vid)].Score).ravel())
        file_fps = probe_video(args.videos / f"{vid}.mp4").fps
        flag = gt_rate_suspect(n_gt, T, args.tol)
        suspects += [vid] if flag else []
        print(f"{vid:30s} {n_gt:9d} {T:7.1f} {n_gt / T:7.2f} {file_fps:8.2f}"
              f"{'  SUSPECT' if flag else ''}")  # fmt: skip
    print(f"\n{len(suspects)}/{len(dur)} suspect: {suspects}")


if __name__ == "__main__":
    main()
