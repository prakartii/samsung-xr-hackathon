"""Compute flow, hands and CLIP signals for every video into the versioned cache.

    python tools/cache_features.py                      # all videos in data/videos
    python tools/cache_features.py --features clip --limit 5

- Longest videos first (PROGRESS.md P3), so the slow ones start early.
- Resumable: entries already in the cache at the current version are skipped
  (features/cache.py), so a stopped run just continues.
- One failing video is logged and skipped; the run goes on.
- Throughput (compute seconds per minute of video) is appended per video and feature
  to <cache>/throughput.csv for PROGRESS.md.

Metadata: ffprobe when it is on PATH (what run.sh uses); otherwise OpenCV's container
metadata, logged as such. Offline tool: it never feeds the filename into features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from rewind.features.cache import FeatureCache, content_fingerprint  # noqa: E402
from rewind.features.clip import (  # noqa: E402
    ClipParams,
    OpenClipEncoder,
    clip_cache_version,
    clip_signals,
)
from rewind.features.flow import FlowParams, flow_cache_version, flow_signals  # noqa: E402
from rewind.features.hands import (  # noqa: E402
    HandParams,
    MediaPipeHands,
    hands_cache_version,
    hands_signals,
)
from rewind.video_io import ProbeMeta, ffprobe_meta, iter_frames  # noqa: E402

FEATURES = ("flow", "hands", "clip")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path: Path) -> tuple[ProbeMeta, str]:
    if shutil.which("ffprobe"):
        return ffprobe_meta(path), "ffprobe"
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if fps <= 0 or n <= 0:
        raise OSError(f"cannot read metadata: {path}")
    return ProbeMeta(duration_sec=n / fps, fps=fps, width=w, height=h), "opencv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--videos", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--cache", type=Path, default=REPO / "cache")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument(
        "--features", default=",".join(FEATURES), help="comma list of " + ",".join(FEATURES)
    )
    ap.add_argument("--limit", type=int, default=None, help="process at most N videos")
    args = ap.parse_args()

    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    if unknown := set(feats) - set(FEATURES):
        ap.error(f"unknown features: {sorted(unknown)}")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dense_fps = cfg["signals"]["dense_fps"]
    clip_fps = cfg["signals"]["clip_fps"]
    flow_p = FlowParams(**cfg["features"]["flow"])
    hand_p = HandParams(**cfg["features"]["hands"])
    clip_p = ClipParams(**cfg["features"]["clip"])

    caches, models = {}, {}
    if "flow" in feats:
        caches["flow"] = FeatureCache(args.cache, "flow", flow_cache_version(flow_p, dense_fps))
    if "hands" in feats:
        v = hands_cache_version(hand_p, dense_fps, sha256_file(REPO / hand_p.model))
        caches["hands"] = FeatureCache(args.cache, "hands", v)
    if "clip" in feats:
        v = clip_cache_version(clip_p, clip_fps, sha256_file(REPO / clip_p.weights))
        caches["clip"] = FeatureCache(args.cache, "clip", v)
        models["clip"] = OpenClipEncoder(clip_p, REPO)

    videos = []
    for p in sorted(args.videos.glob("*.mp4")):
        try:
            meta, source = probe(p)
            videos.append((meta.duration_sec, p, meta, source))
        except OSError as e:
            print(f"skip {p.name}: {e}", flush=True)
    videos.sort(key=lambda v: -v[0])  # longest first
    if args.limit:
        videos = videos[: args.limit]

    log_path = args.cache / "throughput.csv"
    args.cache.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as logf:
        log = csv.writer(logf)
        if new_log:
            log.writerow(
                ["file", "feature", "duration_sec", "compute_sec", "sec_per_video_min", "meta"]
            )
        for i, (duration, path, meta, source) in enumerate(videos, 1):
            fp = content_fingerprint(path)
            print(f"[{i}/{len(videos)}] {path.name} ({duration:.0f} s, meta={source})", flush=True)
            for feat in feats:
                cache = caches[feat]
                if cache.get(fp) is not None:
                    print(f"  {feat}: cached", flush=True)
                    continue

                def compute(feat=feat, path=path, meta=meta):
                    if feat == "flow":
                        frames = iter_frames(
                            path, meta.duration_sec, meta.fps, dense_fps, flow_p.width
                        )
                        return flow_signals(frames, flow_p)
                    if feat == "hands":
                        # A fresh landmarker per video: VIDEO mode needs timestamps
                        # that increase across calls, and each video restarts at 0.
                        with MediaPipeHands(hand_p, REPO) as scorer:
                            frames = iter_frames(path, meta.duration_sec, meta.fps, dense_fps)
                            return hands_signals(frames, scorer)
                    frames = iter_frames(path, meta.duration_sec, meta.fps, clip_fps)
                    return clip_signals(frames, models["clip"], clip_p.batch_size)

                t0 = time.perf_counter()
                try:
                    cache.get_or_compute(fp, compute)
                except Exception:
                    print(f"  {feat}: FAILED\n{traceback.format_exc()}", flush=True)
                    continue
                dt = time.perf_counter() - t0
                per_min = dt / (duration / 60)
                print(f"  {feat}: {dt:.1f} s ({per_min:.1f} s per video minute)", flush=True)
                log.writerow(
                    [path.name, feat, f"{duration:.2f}", f"{dt:.2f}", f"{per_min:.2f}", source]
                )
                logf.flush()


if __name__ == "__main__":
    main()
