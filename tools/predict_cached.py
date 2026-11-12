"""Full pipeline output for a split from the feature cache alone (no MP4 needed).

    python tools/predict_cached.py --split tune --out results/preds/full_tune
    python -m rewind_eval.score --split tune --pred results/preds/full_tune --tag "full tune"

Runs exactly `rewind.analyze.summarize` (route, actions, highlights, label) per video,
reading flow, hands, CLIP and audio from cache/ (audio decoded from the local MP4
when not cached yet) and SlowFast features from cache/ or weights/ by content
fingerprint, as `run.sh` would on a machine where everything is
cached. That lets the `tune` split (EPIC train videos processed on Kaggle and never
stored here) and all of `report` be scored in minutes, which tuning needs (P5, P7).

Fingerprint and duration per id: the Kaggle job logs (cache/kaggle_logs), else the
local MP4 in --videos. Offline evaluation tool: video ids select the cache entries
here, but `summarize` itself never sees anything but signals.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "code"), str(REPO / "tools")]

Index = dict[str, tuple[str, float]]


def video_index(
    ids: list[str], logged: Index, videos: Path, local: Callable[[Path], tuple[str, float]]
) -> tuple[Index, list[str]]:
    """{id: (fingerprint, duration)} from the logs, else from ``local(<videos>/<id>.mp4)``;
    plus the ids with neither."""
    index: Index = {}
    missing = []
    for vid in ids:
        mp4 = videos / f"{vid}.mp4"
        if vid in logged:
            index[vid] = logged[vid]
        elif mp4.exists():
            index[vid] = local(mp4)
        else:
            missing.append(vid)
    return index, missing


def _local(mp4: Path) -> tuple[str, float]:
    from rewind.features.cache import content_fingerprint
    from rewind.video_io import probe_video

    return content_fingerprint(mp4), probe_video(mp4).duration_sec


def split_ids(split: str) -> list[str]:
    """EPIC then FPVSum ids of a split (`tune` FPVSum = all CV-fold videos)."""
    from rewind_eval.splits import load_splits

    s = load_splits()
    epic = s.tune_epic if split == "tune" else s.report_epic
    return [*epic, *sorted(s.fpvsum_folds if split == "tune" else s.report_fpvsum)]


class Predictor:
    """Models loaded once; `run` writes one JSON per video for any config variant that
    uses the same model files. ActionFormer detections are memoized per video (they
    depend on the features and weights only)."""

    def __init__(self, cfg, videos: Path, fingerprint_logs: list[Path]) -> None:
        from import_released_features import load_fingerprint_logs

        from rewind.analyze import load_models

        self.videos = videos
        self.logged = load_fingerprint_logs(fingerprint_logs)
        self.models = load_models(cfg, REPO)
        self._dets: dict[str, object] = {}

    def run(self, cfg, ids: list[str], out: Path, actionformer: bool = True) -> None:
        from rewind.analyze import audio_loader, summarize
        from rewind.features.extract import FEATURES, signal_caches
        from rewind.features.slowfast import find_slowfast
        from rewind.output import dumps, to_document
        from rewind.utils import seed_all

        seed_all(cfg.run.seed)
        index, missing = video_index(ids, self.logged, self.videos, _local)
        if missing:
            raise SystemExit(f"no fingerprint (not in the logs, no MP4): {missing}")
        m = self.models
        caches = signal_caches(m.cache_root, cfg, m.signals)
        text, prior = m.text(cfg), m.prior(cfg)
        roots = [m.cache_root, m.released_root]
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        for i, vid in enumerate(ids, 1):
            fp, duration = index[vid]
            signals = {f: caches[f].get(fp) for f in FEATURES}
            if any(v is None for v in signals.values()):
                raise SystemExit(
                    f"{vid}: not cached: {[f for f, v in signals.items() if v is None]}"
                )

            def detect(vid=vid, fp=fp, duration=duration):
                if not actionformer or m.verb_detector is None or m.noun_detector is None:
                    return None
                if vid not in self._dets:
                    entry = find_slowfast(fp, roots)
                    self._dets[vid] = None if entry is None else (
                        m.verb_detector.detect(entry["feats"], duration),
                        m.noun_detector.detect(entry["feats"], duration),
                    )  # fmt: skip
                return self._dets[vid]

            audio = audio_loader(self.videos / f"{vid}.mp4", fp, cfg, m.cache_root)
            result = summarize(vid, duration, signals, text, prior, cfg, detect, audio)
            (out / f"{vid}.json").write_text(dumps(to_document(result)), encoding="utf-8")
            print(f"[{i}/{len(ids)}] {vid}: {result.label[0]}, {len(result.actions)} actions,"
                  f" {len(result.highlights)} highlights", flush=True)  # fmt: skip
        print(f"{len(ids)} videos in {time.perf_counter() - t0:.0f} s", flush=True)


def default_logs() -> list[Path]:
    return sorted((REPO / "cache" / "kaggle_logs").glob("shard_*.jsonl"))


def main() -> None:
    from rewind.config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=["tune", "report"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument("--videos", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--fingerprint-logs", type=Path, nargs="*", default=default_logs())
    args = ap.parse_args()

    cfg = load_config(args.config)
    Predictor(cfg, args.videos, args.fingerprint_logs).run(cfg, split_ids(args.split), args.out)


if __name__ == "__main__":
    main()
