"""Import ActionFormer's released EPIC SlowFast features into the feature cache.

    python tools/import_released_features.py --features-dir <dir with <video_id>.npz>

e.g. --features-dir D:/rewind-data/actionformer/epic_kitchens/features

For each `<video_id>.npz` (key `feats`, shape (K, 2304)) whose MP4 is in --videos,
store it under the MP4's content fingerprint (ARCHITECTURE.md §8), so inference finds
it without ever using the filename. Offline tool: the filename is only used here to
pair a released file with its MP4.

Also reports (PROGRESS.md P4):
- coverage: EPIC ids in kit/manifest.csv with no released features (to extract on
  Kaggle/Colab), and manifest ids whose MP4 isn't downloaded yet;
- alignment: feature count vs video duration. One feature per 16/30 s, so
  K ≈ (T * 30 - 32) / 16 + 1; a gap over `--max-gap` seconds is reported as suspect.

Re-running is safe: entries already in the cache are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from rewind.detect.actionformer import FEAT_FPS, FEAT_NUM_FRAMES, FEAT_STRIDE  # noqa: E402
from rewind.features.cache import content_fingerprint  # noqa: E402
from rewind.features.slowfast import slowfast_cache, slowfast_entry  # noqa: E402


def expected_feature_count(duration_sec: float) -> int:
    """Clips of 32 frames at 30 fps every 16 frames that fit in the video."""
    frames = int(duration_sec * FEAT_FPS)
    return max(0, (frames - FEAT_NUM_FRAMES) // FEAT_STRIDE + 1)


def alignment_gap_sec(n_feats: int, duration_sec: float) -> float:
    """|K - expected K| in seconds of video (one feature step = 16/30 s)."""
    return abs(n_feats - expected_feature_count(duration_sec)) * FEAT_STRIDE / FEAT_FPS


def load_fingerprint_logs(paths: list[Path]) -> dict[str, tuple[str, float]]:
    """{video_id: (content fingerprint, duration)} from tools/kaggle_cache_epic.py logs.

    Lets features be imported for videos processed on Kaggle and never stored here.
    Failed records (no fingerprint) are skipped. One id with two fingerprints means
    two different files, so it is an error rather than a guess.
    """
    out: dict[str, tuple[str, float]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "fingerprint" not in rec:
                continue
            vid, fp = rec["video_id"], rec["fingerprint"]
            if vid in out and out[vid][0] != fp:
                raise ValueError(f"{vid}: two fingerprints in the logs ({out[vid][0]}, {fp})")
            out[vid] = (fp, float(rec["duration"]))
    return out


def video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()
    return n / fps if fps > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--videos", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--cache", type=Path, default=REPO / "cache")
    ap.add_argument("--max-gap", type=float, default=2.0, help="alignment warning, seconds")
    args = ap.parse_args()

    with (REPO / "kit" / "manifest.csv").open(newline="", encoding="utf-8") as f:
        epic_ids = [r["video_id"] for r in csv.DictReader(f) if r["dataset"] == "epic_kitchens_100"]
    released = {p.stem: p for p in args.features_dir.glob("*.npz")}
    no_features = [v for v in epic_ids if v not in released]
    print(f"released feature files: {len(released)}")
    print(f"coverage: {len(epic_ids) - len(no_features)}/{len(epic_ids)} manifest EPIC videos")
    for v in no_features:
        print(f"  NO RELEASED FEATURES: {v}")

    cache = slowfast_cache(args.cache)
    imported = skipped = not_downloaded = suspect = 0
    for vid in epic_ids:
        if vid not in released:
            continue
        mp4 = args.videos / f"{vid}.mp4"
        if not mp4.exists():
            not_downloaded += 1
            continue
        fp = content_fingerprint(mp4)
        if cache.get(fp) is not None:
            skipped += 1
            continue
        with np.load(released[vid]) as npz:
            feats = npz["feats"]
        gap = alignment_gap_sec(len(feats), video_duration(mp4))
        if not gap <= args.max_gap:
            suspect += 1
            dur = video_duration(mp4)
            print(f"  ALIGNMENT? {vid}: {len(feats)} feats vs {dur:.1f} s (gap {gap:.1f} s)")
        cache.put(fp, slowfast_entry(feats))
        imported += 1
    print(
        f"imported {imported}, already cached {skipped}, MP4 not downloaded yet {not_downloaded},"
        f" alignment warnings {suspect}"
    )


if __name__ == "__main__":
    main()
