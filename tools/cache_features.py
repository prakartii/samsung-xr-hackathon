"""Compute flow, hands and CLIP signals for every video into the versioned cache.

    python tools/cache_features.py                      # all videos in data/videos
    python tools/cache_features.py --features clip --limit 5
    python tools/cache_features.py --shard 0/2 --torch-threads 6   # + --shard 1/2

- One decoding pass per video feeds every missing feature (flow, hands, CLIP).
- Longest videos first (PROGRESS.md P3), so the slow ones start early.
- Resumable: entries already in the cache at the current version are skipped
  (features/cache.py), so a stopped run just continues.
- One failing video is logged and skipped; the run goes on.
- `--shard i/n` takes every n-th video (round-robin over the longest-first order,
  so shards get similar loads); run n processes side by side to use all cores.
- Throughput (compute seconds per minute of video) is appended per video and feature
  to <cache>/throughput/<features>_<shard>.csv (one file per process) for PROGRESS.md.

Metadata and decoding: PyAV (rewind.video_io). Offline tool: it never feeds the
filename into features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from rewind.config import load_config  # noqa: E402
from rewind.features.cache import FeatureCache, content_fingerprint  # noqa: E402
from rewind.features.clip import (  # noqa: E402
    ClipOnline,
    ClipParams,
    OpenClipEncoder,
    clip_cache_version,
)
from rewind.features.flow import FlowOnline, FlowParams, flow_cache_version  # noqa: E402
from rewind.features.hands import (  # noqa: E402
    HandParams,
    HandsOnline,
    MediaPipeHands,
    hands_cache_version,
)
from rewind.video_io import iter_frames, probe_video  # noqa: E402

FEATURES = ("flow", "hands", "clip")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--videos", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--cache", type=Path, default=REPO / "cache")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument(
        "--features", default=",".join(FEATURES), help="comma list of " + ",".join(FEATURES)
    )
    ap.add_argument("--limit", type=int, default=None, help="process at most N videos")
    ap.add_argument("--shard", default="0/1", help="i/n: process videos i, i+n, i+2n, ...")
    ap.add_argument("--torch-threads", type=int, default=None, help="CPU threads for CLIP")
    args = ap.parse_args()

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    if not 0 <= shard_i < shard_n:
        ap.error("--shard must be i/n with 0 <= i < n")
    if args.torch_threads:
        import torch

        torch.set_num_threads(args.torch_threads)

    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    if unknown := set(feats) - set(FEATURES):
        ap.error(f"unknown features: {sorted(unknown)}")

    cfg = load_config(args.config)  # strict: unknown or missing keys are errors
    dense_fps = cfg.signals.dense_fps
    clip_fps = cfg.signals.clip_fps
    flow_p = FlowParams(**asdict(cfg.features.flow))
    hand_p = HandParams(**asdict(cfg.features.hands))
    clip_p = ClipParams(**asdict(cfg.features.clip))

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
            meta, source = probe_video(p), "pyav"
            videos.append((meta.duration_sec, p, meta, source))
        except OSError as e:
            print(f"skip {p.name}: {e}", flush=True)
    videos.sort(key=lambda v: -v[0])  # longest first
    videos = videos[shard_i::shard_n]
    if args.limit:
        videos = videos[: args.limit]

    cfg_bits = (dense_fps, clip_fps, flow_p, hand_p, clip_p)
    log_path = args.cache / "throughput" / f"{'-'.join(feats)}_{shard_i}of{shard_n}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
            missing = [f for f in feats if caches[f].get(fp) is None]
            if not missing:
                print("  all cached", flush=True)
                continue
            t0 = time.perf_counter()
            try:
                results = compute_single_pass(path, meta.duration_sec, missing, models, cfg_bits)
            except Exception:
                failed = "+".join(missing)
                print(f"  {failed}: FAILED", traceback.format_exc(), sep="\n", flush=True)
                continue
            for feat, arrays in results.items():
                caches[feat].put(fp, arrays)
            dt = time.perf_counter() - t0
            per_min = dt / (duration / 60)
            label = "+".join(missing)
            print(f"  {label}: {dt:.1f} s ({per_min:.1f} s per video minute)", flush=True)
            log.writerow(
                [path.name, label, f"{duration:.2f}", f"{dt:.2f}", f"{per_min:.2f}", source]
            )
            logf.flush()


def compute_single_pass(path, duration, missing, models, cfg_bits) -> dict:
    """Decode once and feed every missing feature (ARCHITECTURE.md §12 budget).

    Flow and hands use the dense grid; CLIP uses every r-th dense sample, where
    r = dense_fps / clip_fps, with its own timestamps j / clip_fps (identical to
    sample_times(T, clip_fps)). Frames are decoded at the widest size any feature
    needs; each feature downsizes from there.
    """
    dense_fps, clip_fps, flow_p, hand_p, clip_p = cfg_bits
    need_dense = bool({"flow", "hands"} & set(missing))
    ratio = dense_fps / clip_fps
    if need_dense and abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("signals.dense_fps must be a multiple of signals.clip_fps")
    rate, step = (dense_fps, round(ratio)) if need_dense else (clip_fps, 1)
    width = max(
        [flow_p.width] * ("flow" in missing)
        + [hand_p.width] * ("hands" in missing)
        + [clip_p.width] * ("clip" in missing)
    )
    online = {}
    hands_model = None
    if "flow" in missing:
        online["flow"] = FlowOnline(flow_p)
    if "hands" in missing:
        # A fresh landmarker per video: VIDEO mode needs increasing timestamps.
        hands_model = MediaPipeHands(hand_p, REPO)
        online["hands"] = HandsOnline(hands_model)
    if "clip" in missing:
        online["clip"] = ClipOnline(models["clip"], clip_p.batch_size)
    try:
        for k, (t, frame) in enumerate(iter_frames(path, duration, rate, width)):
            if "flow" in online:
                online["flow"].add(t, frame)
            if "hands" in online:
                online["hands"].add(t, frame)
            if "clip" in online and k % step == 0:
                online["clip"].add((k // step) / clip_fps, frame)
    finally:
        if hands_model is not None:
            hands_model.close()
    return {feat: o.result() for feat, o in online.items()}


if __name__ == "__main__":
    main()
