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
4. Write /kaggle/working/shard_<i>.jsonl: video_id, fingerprint, duration, seconds, and the
   versions of the feature libraries (installed at the requirements.txt pins).

The output folder (/kaggle/working) holds only the cache (~100 MB for all of EPIC);
tools/kaggle_launch.py --collect downloads and merges it into the local cache.
Entries are keyed by content fingerprint, exactly as a local run would key them.
"""

import json
import queue
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SHARD = globals().get("SHARD", 0)
N_SHARDS = globals().get("N_SHARDS", 1)
# Retry pass: only these EPIC ids (set by kaggle_launch.py --only-ids), else the manifest.
ONLY_IDS = globals().get("ONLY_IDS")
DOWNLOAD_THREADS = 12
BACKLOG = 4
WORK = Path("/kaggle/working")
TMP = Path("/tmp/rewind")


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def pinned(requirements: Path, names: list[str]) -> list[str]:
    """`name==version` lines for ``names`` from a requirements file (comments ignored).

    The Kaggle job installs exactly what run.sh ships, so cached features can't come
    from a different decoder or model version than the one used at inference.
    """
    found = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        spec = line.split("#", 1)[0].strip()
        if "==" in spec:
            name = spec.split("==", 1)[0].strip()
            found[name.lower()] = spec
    missing = [n for n in names if n.lower() not in found]
    if missing:
        raise ValueError(f"{requirements}: no pin for {missing}")
    return [found[n.lower()] for n in names]


def setup() -> Path:
    repo = TMP / "repo"
    # Kaggle auto-extracts an uploaded .tar.gz into a read-only folder; older datasets
    # may keep the archive. Accept both, and copy to a writable place (weights land
    # in repo/weights).
    tars = list(Path("/kaggle/input").rglob("rewind-code.tar.gz"))
    if tars and tars[0].is_file():
        repo.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tars[0]) as tf:
            tf.extractall(repo)
    else:
        marker = next(Path("/kaggle/input").rglob("configs/default.yaml"))
        shutil.copytree(marker.parents[1], repo, dirs_exist_ok=True)
    # Same versions as requirements.txt. mediapipe pins opencv-contrib-python by name;
    # the image's OpenCV already provides cv2, so install mediapipe without deps plus
    # its small pure deps.
    av, open_clip, mediapipe = pinned(
        repo / "requirements.txt", ["av", "open_clip_torch", "mediapipe"]
    )
    sh(f"pip install -q {av} {open_clip}")
    sh(f"pip install -q --no-deps {mediapipe}")
    sh('pip install -q "absl-py~=2.3" "flatbuffers~=25.9" "sounddevice~=0.5"')
    sh(f"cd {repo} && SKIP_GDRIVE=1 PYTHON={sys.executable} bash tools/fetch_weights.sh")
    return repo


def fetch_and_report(vid: str, download, ready: queue.Queue) -> None:
    """Run ``download(vid)`` and always put (vid, ok) on ``ready``.

    The consumer blocks on ready.get() once per video, so a download that raised without
    putting would hang the shard until Kaggle's 12 h limit (e.g. http.client.
    IncompleteRead, which isn't an OSError and isn't caught downstream).
    """
    ok = False
    try:
        ok = bool(download(vid))
    except Exception as e:  # never leave the consumer waiting
        print(f"[{vid}] fetch crashed: {e!r}", flush=True)
    finally:
        ready.put((vid, ok))


def library_versions() -> dict[str, str]:
    """Versions of the libraries that produce features, stored with every record so any
    Kaggle-vs-local skew can be traced (cache versions don't encode them)."""
    import importlib

    out = {}
    for mod in ("av", "cv2", "torch", "open_clip", "mediapipe", "numpy"):
        try:
            out[mod] = str(importlib.import_module(mod).__version__)
        except Exception as e:  # report, don't fail the shard over a version string
            out[mod] = f"unavailable: {e!r}"
    return out


def main() -> None:
    repo = setup()
    sys.path[:0] = [str(repo / "code"), str(repo / "tools")]
    import cache_features as cf
    import download_videos as dv
    import yaml

    from rewind.features.cache import FeatureCache, content_fingerprint
    from rewind.features.clip import ClipParams, OpenClipEncoder, clip_cache_version
    from rewind.features.flow import FlowParams, flow_cache_version
    from rewind.features.hands import HandParams, hands_cache_version
    from rewind.video_io import probe_video

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
    pool = [v for v in manifest if v in set(ONLY_IDS)] if ONLY_IDS else list(manifest)
    ids = sorted(pool, key=lambda v: -manifest[v])[SHARD::N_SHARDS]
    data = repo / "epic_download_data"
    splits = {r["video_id"]: r["split"] for r in dv._read_csv(data / "epic_55_splits.csv")}
    md5_rows = dv._read_csv(data / "md5.csv")
    md5s = {(r["version"], r["file_remote_path"]): r["md5"] for r in md5_rows}
    minutes = sum(manifest[v] for v in ids) / 60
    print(f"shard {SHARD}/{N_SHARDS}: {len(ids)} videos, {minutes:.0f} min", flush=True)

    ready: queue.Queue = queue.Queue()
    slots = threading.Semaphore(BACKLOG)

    def download(vid: str) -> bool:
        base, parts, version = dv.epic_remote(vid, splits)
        return dv.download_one_epic(
            vid, f"[{vid}]", "/".join([base, *parts]),
            md5s.get((version, "/".join(parts))), TMP, 8,
        )  # fmt: skip

    def fetch(vid: str) -> None:
        slots.acquire()
        fetch_and_report(vid, download, ready)

    libs = library_versions()
    print("libraries:", libs, flush=True)
    tag = "_retry" if ONLY_IDS else ""  # keeps the first pass's log when collected
    log = (WORK / f"shard_{SHARD}{tag}.jsonl").open("a", encoding="utf-8")
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
                       "seconds": round(time.time() - t0, 1), "libs": libs}  # fmt: skip
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


# Runs as a Kaggle script kernel; importing it elsewhere (tests) must not start a shard.
if __name__ == "__main__" or Path("/kaggle/working").exists():
    main()
