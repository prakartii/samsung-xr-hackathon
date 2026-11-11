"""Runs ON KAGGLE: download one shard of the EPIC manifest videos and cache their features.

Launched by tools/kaggle_launch.py, which uploads the repo code as a private dataset
(`rewind-code`, a tar.gz) and pushes one copy of this script per shard with
`SHARD = i` / `N_SHARDS = n` prepended. Home bandwidth (~2.5 MiB/s) would need ~20 h
for the 184 GB; Kaggle's link lets each shard run many 0.5 MiB/s connections.

Per shard:
1. Install runtime deps missing on the Kaggle image; unpack the code; fetch the hands
   + CLIP weights (SHA-256 checked, tools/fetch_weights.sh with SKIP_GDRIVE=1).
2. Download videos (longest first) with `download_threads` parallel connections,
   MD5-verified by tools/download_videos.py, keeping at most `backlog` videos on disk.
3. For each video: content fingerprint, flow + hands + CLIP in one decoding pass
   (tools/cache_features.py), store in /kaggle/working/cache, delete the MP4.
4. Write /kaggle/working/shard_<i>.jsonl: video_id, fingerprint, duration, seconds.

The output folder (/kaggle/working) holds only the cache (~100 MB for all of EPIC);
tools/kaggle_launch.py --collect downloads and merges it into the local cache.
Entries are keyed by content fingerprint, exactly as a local run would key them.
"""

import json
import queue
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SHARD = globals().get("SHARD", 0)
N_SHARDS = globals().get("N_SHARDS", 1)
DOWNLOAD_THREADS = 12
BACKLOG = 4
WORK = Path("/kaggle/working")
TMP = Path("/tmp/rewind")


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def setup() -> Path:
    # mediapipe pins opencv-contrib-python by name; the image's OpenCV already provides
    # cv2, so install mediapipe without deps plus its small pure deps.
    sh("pip install -q av==19.0.2 open_clip_torch==4.2.2")
    sh("pip install -q --no-deps mediapipe==1.2.2")
    sh("pip install -q absl-py flatbuffers sounddevice")
    code_tar = next(Path("/kaggle/input").rglob("rewind-code.tar.gz"))
    repo = TMP / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    with tarfile.open(code_tar) as tf:
        tf.extractall(repo)
    sh(f"cd {repo} && SKIP_GDRIVE=1 PYTHON={sys.executable} bash tools/fetch_weights.sh")
    return repo


def main() -> None:
    repo = setup()
    sys.path[:0] = [str(repo / "code"), str(repo / "tools")]
    import cache_features as cf  # noqa: E402  (tools/, after setup)
    import download_videos as dv  # noqa: E402
    import yaml  # noqa: E402

    from rewind.features.cache import FeatureCache, content_fingerprint  # noqa: E402
    from rewind.features.clip import ClipParams, OpenClipEncoder, clip_cache_version  # noqa: E402
    from rewind.features.flow import FlowParams, flow_cache_version  # noqa: E402
    from rewind.features.hands import HandParams, hands_cache_version  # noqa: E402
    from rewind.video_io import probe_video  # noqa: E402

    cfg = yaml.safe_load((repo / "configs" / "default.yaml").read_text(encoding="utf-8"))
    dense, clip_fps = cfg["signals"]["dense_fps"], cfg["signals"]["clip_fps"]
    flow_p = FlowParams(**cfg["features"]["flow"])
    hand_p = HandParams(**cfg["features"]["hands"])
    clip_p = ClipParams(**cfg["features"]["clip"])
    cache_root = WORK / "cache"
    caches = {
        "flow": FeatureCache(cache_root, "flow", flow_cache_version(flow_p, dense)),
        "hands": FeatureCache(
            cache_root, "hands",
            hands_cache_version(hand_p, dense, cf.sha256_file(repo / hand_p.model)),
        ),
        "clip": FeatureCache(
            cache_root, "clip",
            clip_cache_version(clip_p, clip_fps, cf.sha256_file(repo / clip_p.weights)),
        ),
    }  # fmt: skip
    models = {"clip": OpenClipEncoder(clip_p, repo)}
    cfg_bits = (dense, clip_fps, flow_p, hand_p, clip_p)

    manifest = dv.read_manifest(repo / "kit" / "manifest.csv")["epic_kitchens_100"]
    ids = sorted(manifest, key=lambda v: -manifest[v])[SHARD::N_SHARDS]
    data = repo / "epic_download_data"
    splits = {r["video_id"]: r["split"] for r in dv._read_csv(data / "epic_55_splits.csv")}
    md5_rows = dv._read_csv(data / "md5.csv")
    md5s = {(r["version"], r["file_remote_path"]): r["md5"] for r in md5_rows}
    minutes = sum(manifest[v] for v in ids) / 60
    print(f"shard {SHARD}/{N_SHARDS}: {len(ids)} videos, {minutes:.0f} min", flush=True)

    ready: queue.Queue = queue.Queue()
    slots = threading.Semaphore(BACKLOG)

    def fetch(vid: str) -> None:
        slots.acquire()
        base, parts, version = dv.epic_remote(vid, splits)
        ok = dv.download_one_epic(
            vid, f"[{vid}]", "/".join([base, *parts]), md5s.get((version, "/".join(parts))), TMP, 8
        )
        ready.put((vid, ok))

    log = (WORK / f"shard_{SHARD}.jsonl").open("a", encoding="utf-8")
    with ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
        for vid in ids:
            pool.submit(fetch, vid)
        for _ in ids:
            vid, ok = ready.get()
            mp4 = dv.epic_target(TMP, vid)
            try:
                if not ok:
                    raise RuntimeError("download failed")
                t0 = time.time()
                fp = content_fingerprint(mp4)
                meta = probe_video(mp4)
                missing = [f for f in caches if caches[f].get(fp) is None]
                for feat, arrays in cf.compute_single_pass(
                    mp4, meta.duration_sec, missing, models, cfg_bits
                ).items():
                    caches[feat].put(fp, arrays)
                rec = {"video_id": vid, "fingerprint": fp, "duration": meta.duration_sec,
                       "seconds": round(time.time() - t0, 1)}  # fmt: skip
                print("done", rec, flush=True)
            except Exception as e:  # keep going; the record says what failed
                rec = {"video_id": vid, "error": repr(e)}
                print("FAILED", rec, flush=True)
            finally:
                for p in mp4.parent.glob(f"{vid}.MP4*"):
                    p.unlink(missing_ok=True)
                slots.release()
            log.write(json.dumps(rec) + "\n")
            log.flush()


main()
