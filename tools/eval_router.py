"""Measure routing agreement with kit/manifest.csv's `dataset` column (P6, ARCHITECTURE §6.1).

    python tools/eval_router.py

For every manifest video: its cached 1 fps CLIP embeddings (EPIC fingerprints from
the Kaggle job logs, FPVSum from the local MP4), `label.n_keyframes` evenly spaced
keyframes, zero-shot P(cooking) over `label.classes` with `label.prompts`, then
`rewind.routing.route` with `routing.cooking_thr` / `routing.min_conf`. Reports the
agreement (target >= 95%), the misrouted ids and the uncertain ones.

Measurement only: the manifest videos are the `report` split, which is never used
for tuning (CLAUDE.md §6.2). Offline tool; inference never reads the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "code"), str(REPO / "tools")]

from rewind.config import load_config  # noqa: E402
from rewind.features.cache import FeatureCache, content_fingerprint  # noqa: E402
from rewind.features.clip import (  # noqa: E402
    ClipParams,
    OpenClipEncoder,
    class_embeddings,
    clip_cache_version,
    zero_shot_probs,
)
from rewind.label import keyframe_indices  # noqa: E402
from rewind.routing import route  # noqa: E402


def agreement(routed: dict[str, str], truth: dict[str, str]) -> dict:
    """Share of videos routed to their true dataset; `unknown` counts as not agreeing."""
    wrong = sorted(v for v, d in routed.items() if d not in ("unknown", truth[v]))
    uncertain = sorted(v for v, d in routed.items() if d == "unknown")
    correct = sum(1 for v, d in routed.items() if d == truth[v])
    return {"agreement": correct / len(routed), "wrong": wrong, "uncertain": uncertain}


def main() -> None:
    from import_released_features import load_fingerprint_logs

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument("--videos", type=Path, default=REPO / "data" / "videos")
    ap.add_argument(
        "--fingerprint-logs",
        type=Path,
        nargs="*",
        default=sorted((REPO / "cache" / "kaggle_logs").glob("shard_*.jsonl")),
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    clip_p = ClipParams(**asdict(cfg.features.clip))
    weights_sha = hashlib.sha256((REPO / clip_p.weights).read_bytes()).hexdigest()
    cache = FeatureCache(
        REPO / cfg.features.cache_dir,
        "clip",
        clip_cache_version(clip_p, cfg.signals.clip_fps, weights_sha),
    )
    logged = load_fingerprint_logs(args.fingerprint_logs)
    encoder = OpenClipEncoder(clip_p, REPO)
    classes = dict(cfg.label.classes)
    cls_emb = class_embeddings(encoder, classes, cfg.label.prompts)
    cooking = list(classes).index("cooking")

    with (REPO / "kit" / "manifest.csv").open(newline="", encoding="utf-8") as f:
        truth = {r["video_id"]: r["dataset"] for r in csv.DictReader(f)}
    routed, p_cook = {}, {}
    for vid in truth:
        mp4 = args.videos / f"{vid}.mp4"
        fp = content_fingerprint(mp4) if mp4.exists() else logged[vid][0]
        entry = cache.get(fp)
        if entry is None:
            raise SystemExit(f"{vid}: no CLIP cache entry")
        emb = entry["clip_emb"][keyframe_indices(len(entry["clip_emb"]), cfg.label.n_keyframes)]
        probs = zero_shot_probs(emb, cls_emb, encoder.logit_scale)[:, cooking]
        decision = route(probs, cfg.routing.cooking_thr, cfg.routing.min_conf)
        routed[vid], p_cook[vid] = decision.dataset, decision.p_cooking

    stats = agreement(routed, truth)
    for ds in ("epic_kitchens_100", "fpvsum"):
        vals = np.array([p_cook[v] for v in truth if truth[v] == ds])
        print(f"{ds}: n={len(vals)}, P(cooking) min {vals.min():.3f} / median "
              f"{np.median(vals):.3f} / max {vals.max():.3f}")  # fmt: skip
    print(json.dumps({**stats, "n": len(routed)}, indent=1))


if __name__ == "__main__":
    main()
