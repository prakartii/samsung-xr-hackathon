"""Tune the FPVSum route on the CV folds (P6): importance weights or label phrases.

    python tools/tune_fpvsum.py --what importance
    python tools/tune_fpvsum.py --what label

FPVSum has no separate tuning videos, so `tune` is 5-fold cross-validation over
the 49 videos (`configs/splits.yaml` -> `tune.fpvsum_folds`; CLAUDE.md §6.2). Each
candidate config is scored once per video (frame F1 for importance, 0/1 accuracy
for the label); then for each fold the config that is best on the other 4 folds is
scored on the held-out one. The mean held-out score is the honest estimate; the
config best on all folds is the one to adopt. The search grids are fixed below,
before looking at any result.

Uses the cache (`tools/cache_features.py`) and the eval ground truth; offline only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from rewind.config import Config, load_config  # noqa: E402
from rewind.features.cache import FeatureCache, content_fingerprint  # noqa: E402
from rewind.features.clip import (  # noqa: E402
    ClipParams,
    OpenClipEncoder,
    class_embeddings,
    clip_cache_version,
)
from rewind.features.flow import FlowParams, flow_cache_version  # noqa: E402
from rewind.features.hands import HandParams, hands_cache_version  # noqa: E402
from rewind.label import fpvsum_label  # noqa: E402
from rewind.postprocess import postprocess  # noqa: E402
from rewind.select.importance import ImportanceParams, fpvsum_segments  # noqa: E402
from rewind_eval.f1 import frame_f1  # noqa: E402
from rewind_eval.gt_fpvsum import load_fpvsum  # noqa: E402
from rewind_eval.splits import load_splits  # noqa: E402

# Importance grid (fixed in advance): each weight in {0, 0.5, 1}, sigma in {1, 2, 4} s.
WEIGHTS = (0.0, 0.5, 1.0)
SIGMAS = (1.0, 2.0, 4.0)

# Label phrase alternatives for the classes confused in the first run (P6 result):
# jumping -> plane (3), bikepolo -> bike (2), bike -> motor, longboarding -> motor,
# plane -> horse. Other classes keep their config phrase.
PHRASES = {
    "jumping": ("skydiving or jumping", "skydiving in free fall", "base jumping off a cliff"),
    "bikepolo": ("playing bike polo", "playing polo on bicycles with mallets"),
    "bike": ("riding a bicycle", "mountain biking on a trail"),
    "longboarding": ("riding a longboard", "longboard skateboarding down a road"),
    "plane": ("flying a small plane", "the cockpit view of a flying airplane"),
}


def cross_validate(scores: Mapping[str, Mapping[str, float]], folds: Mapping[str, int]) -> dict:
    """Per held-out fold: the config best on the other folds and its held-out mean.

    `scores[config][video]`; configs are compared in their given order (ties keep
    the earlier one). `chosen` is the config best on all videos.
    """
    configs = list(scores)

    def mean(cfg: str, vids: list[str]) -> float:
        return float(np.mean([scores[cfg][v] for v in vids]))

    def best(vids: list[str]) -> str:
        return max(configs, key=lambda c: (mean(c, vids), -configs.index(c)))

    per_fold = {}
    for k in sorted(set(folds.values())):
        held = [v for v, f in folds.items() if f == k]
        train = [v for v, f in folds.items() if f != k]
        pick = best(train)
        per_fold[k] = (pick, mean(pick, held))
    return {
        "per_fold": per_fold,
        "cv_mean": float(np.mean([s for _, s in per_fold.values()])),
        "chosen": best(list(folds)),
    }


def importance_grid(cfg: Config) -> dict[str, ImportanceParams]:
    base = ImportanceParams(**asdict(cfg.importance))
    grid = {"current": base}
    for wf, wh, wg, ws, sig in itertools.product(WEIGHTS, WEIGHTS, WEIGHTS, WEIGHTS, SIGMAS):
        if wf == wh == wg == ws == 0.0:
            continue  # no signal at all
        name = f"flow={wf} hands={wh} global={wg} scene={ws} sigma={sig}"
        grid[name] = replace(
            base, w_flow=wf, w_hands=wh, w_global_motion=wg, w_scene_change=ws, sigma=sig
        )
    return grid


def label_grid(cfg: Config) -> dict[str, dict[str, str]]:
    base = {k: v for k, v in cfg.label.classes.items() if k != "cooking"}
    grid = {"current": base}
    keys = list(PHRASES)
    for choice in itertools.product(*(PHRASES[k] for k in keys)):
        phrases = dict(base, **dict(zip(keys, choice, strict=True)))
        grid[" | ".join(choice)] = phrases
    return grid


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--what", choices=["importance", "label"], required=True)
    ap.add_argument("--videos", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    folds = load_splits().fpvsum_folds
    dense, clip_fps = cfg.signals.dense_fps, cfg.signals.clip_fps
    hand_p = HandParams(**asdict(cfg.features.hands))
    clip_p = ClipParams(**asdict(cfg.features.clip))
    root = REPO / cfg.features.cache_dir
    caches = {
        "flow": FeatureCache(
            root, "flow", flow_cache_version(FlowParams(**asdict(cfg.features.flow)), dense)
        ),
        "hands": FeatureCache(
            root, "hands", hands_cache_version(hand_p, dense, _sha256(REPO / hand_p.model))
        ),
        "clip": FeatureCache(
            root, "clip", clip_cache_version(clip_p, clip_fps, _sha256(REPO / clip_p.weights))
        ),
    }
    import csv

    with (REPO / "kit" / "manifest.csv").open(newline="", encoding="utf-8") as f:
        durations = {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader(f)}
    gt = load_fpvsum(
        REPO / "data" / "annotations" / "fpvsum" / "gt_data.mat",
        {v: durations[v] for v in folds},
    )
    videos = {}
    for vid in folds:
        fp = content_fingerprint(args.videos / f"{vid}.mp4")
        videos[vid] = {name: cache.get(fp) for name, cache in caches.items()}

    scores: dict[str, dict[str, float]] = {}
    if args.what == "importance":
        for name, imp in importance_grid(cfg).items():
            scores[name] = {}
            for vid, sig in videos.items():
                T = durations[vid]
                segs = fpvsum_segments(
                    sig["flow"], sig["hands"], sig["clip"], T, cfg.select.budget_frac, imp
                )
                hl = postprocess(
                    segs, [0.0] * len(segs), T, budget_frac=cfg.select.budget_frac,
                    merge_gap=cfg.post.merge_gap, short_video_mode=cfg.post.short_video_mode,
                )  # fmt: skip
                scores[name][vid] = frame_f1(gt[vid].times, gt[vid].highlight, hl)
    else:
        encoder = OpenClipEncoder(clip_p, REPO)
        for name, phrases in label_grid(cfg).items():
            emb = class_embeddings(encoder, phrases, cfg.label.prompts)
            keys = list(phrases)
            scores[name] = {}
            for vid, sig in videos.items():
                name_pred, _ = fpvsum_label(
                    sig["clip"]["clip_emb"], emb, keys, encoder.logit_scale, cfg.label.n_keyframes
                )
                scores[name][vid] = float(name_pred == gt[vid].label)

    cv = cross_validate(scores, folds)
    current = cross_validate({"current": scores["current"]}, folds)
    print(json.dumps({
        "what": args.what,
        "n_configs": len(scores),
        "current_cv_mean": round(current["cv_mean"], 4),
        "tuned_cv_mean": round(cv["cv_mean"], 4),
        "per_fold": {k: [c, round(s, 4)] for k, (c, s) in cv["per_fold"].items()},
        "chosen": cv["chosen"],
        "chosen_all_folds_mean": round(float(np.mean(list(scores[cv["chosen"]].values()))), 4),
    }, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
