"""Predict EPIC actions[] for a split from the released SlowFast features, for scoring.

    python tools/predict_epic_actions.py --split tune --out results/preds/af_tune
    python -m rewind_eval.score --split tune --pred results/preds/af_tune --tag "AF tune"

Offline evaluation tool (P4 exit check: mAP on `tune` and `report`). It works per
video id on ActionFormer's released features (`<features-dir>/<id>.npz`), so the
`tune` split (EPIC *train* videos) needs no video download. Inference never does
this; it reads only the MP4 and the fingerprint-keyed cache.

Each document is valid in the kit schema: actions from ActionFormer (verb + noun
models) paired by `detect/pairing.py`; highlights from the uniform fallback and the
`cooking` label, since mAP only scores actions. Duration comes from
EPIC_100_video_info.csv (data/annotations, tools/fetch_annotations.sh).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "code"), str(REPO / "tools")]

from rewind.config import Config, load_config  # noqa: E402
from rewind.detect.actionformer import ActionFormerDetector, ActionFormerParams  # noqa: E402
from rewind.detect.pairing import (  # noqa: E402
    Action,
    PairingParams,
    VNPrior,
    pair_proposals,
    proposals_to_actions,
)
from rewind.fallback import fallback_result  # noqa: E402
from rewind.output import dumps, to_document  # noqa: E402
from rewind.vocab import load_vocab  # noqa: E402
from rewind_eval.gt_epic import load_video_durations  # noqa: E402
from rewind_eval.splits import load_splits  # noqa: E402

AF_CONFIGS = REPO / "code" / "third_party" / "actionformer" / "configs"


def epic_document(video_id: str, duration: float, actions: list[Action], cfg: Config) -> dict:
    """Kit document: the given actions, fallback highlights, label `cooking`."""
    base = fallback_result(
        cfg.team.team_id,
        video_id,
        duration,
        budget_frac=cfg.select.budget_frac,
        merge_gap=cfg.post.merge_gap,
        short_video_mode=cfg.post.short_video_mode,
    )
    return to_document(replace(base, label=("cooking", 1.0), actions=tuple(actions)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=["tune", "report"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--features-dir",
        type=Path,
        default=REPO / "data" / "actionformer" / "epic_kitchens" / "features",
    )
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument(
        "--detector",
        choices=["actionformer", "zeroshot"],
        default="actionformer",
        help="zeroshot: CLIP sliding windows (P4 kill-rule baseline); report split only",
    )
    ap.add_argument(
        "--fingerprint-logs",
        type=Path,
        nargs="*",
        default=sorted((REPO / "cache" / "kaggle_logs").glob("shard_*.jsonl")),
        help="zeroshot: Kaggle job logs mapping ids to cache fingerprints",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    splits = load_splits()
    ids = list(splits.tune_epic if args.split == "tune" else splits.report_epic)
    durations = load_video_durations(
        REPO / "data" / "annotations" / "epic" / "EPIC_100_video_info.csv"
    )
    vocab = load_vocab()
    verbs, nouns = list(vocab.verb_keys), list(vocab.noun_keys)
    if args.detector == "zeroshot":
        detect = zeroshot_detector(cfg, verbs, nouns, args.fingerprint_logs)
    else:
        detect = actionformer_detector(verbs, nouns, args.features_dir)
    prior = VNPrior.from_counts(
        verbs, nouns, np.load(REPO / "weights" / "vn_prior.npy"), cfg.pairing.eps
    )
    params = PairingParams(**asdict(cfg.pairing))

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for i, vid in enumerate(ids, 1):
        duration = durations[vid]
        v_dets, n_dets = detect(vid, duration)
        proposals = pair_proposals(v_dets, n_dets, prior, set(verbs), set(nouns), params)
        actions = proposals_to_actions(proposals, cfg.actions.max_per_video, cfg.actions.min_score)
        doc = epic_document(vid, duration, actions, cfg)
        (args.out / f"{vid}.json").write_text(dumps(doc), encoding="utf-8")
        print(f"[{i}/{len(ids)}] {vid}: {len(actions)} actions", flush=True)
    elapsed = time.perf_counter() - t0
    minutes = sum(durations[v] for v in ids) / 60
    print(json.dumps({"videos": len(ids), "seconds": round(elapsed, 1),
                      "sec_per_video_min": round(elapsed / minutes, 2)}))  # fmt: skip


def actionformer_detector(verbs, nouns, features_dir: Path):
    """(video_id, duration) -> (verb dets, noun dets) from released SlowFast features."""
    detectors = {
        kind: ActionFormerDetector(
            ActionFormerParams(
                str(AF_CONFIGS / f"epic_slowfast_{kind}.yaml"),
                f"weights/actionformer_epic_{kind}.pth.tar",
            ),
            keys,
            REPO,
        )
        for kind, keys in (("verb", verbs), ("noun", nouns))
    }

    def detect(vid: str, duration: float):
        with np.load(features_dir / f"{vid}.npz") as npz:
            feats = npz["feats"]
        return detectors["verb"].detect(feats, duration), detectors["noun"].detect(feats, duration)

    return detect


def zeroshot_detector(cfg: Config, verbs, nouns, fingerprint_logs: list[Path]):
    """(video_id, duration) -> (verb dets, noun dets) from cached 1 fps CLIP embeddings."""
    from import_released_features import load_fingerprint_logs

    from rewind.detect.fallback_zeroshot import ZeroShotParams, class_phrase, detect_zeroshot
    from rewind.features.cache import FeatureCache
    from rewind.features.clip import (
        ClipParams,
        OpenClipEncoder,
        class_embeddings,
        clip_cache_version,
    )

    clip_p = ClipParams(**asdict(cfg.features.clip))
    weights_sha = hashlib.sha256((REPO / clip_p.weights).read_bytes()).hexdigest()
    cache = FeatureCache(
        REPO / cfg.features.cache_dir,
        "clip",
        clip_cache_version(clip_p, cfg.signals.clip_fps, weights_sha),
    )
    fingerprints = load_fingerprint_logs(fingerprint_logs)
    encoder = OpenClipEncoder(clip_p, REPO)
    zp = ZeroShotParams(**asdict(cfg.zeroshot))
    verb_text = class_embeddings(encoder, {k: class_phrase(k) for k in verbs}, [zp.verb_template])
    noun_text = class_embeddings(encoder, {k: class_phrase(k) for k in nouns}, [zp.noun_template])

    def detect(vid: str, duration: float):
        entry = cache.get(fingerprints[vid][0])
        if entry is None:
            raise SystemExit(f"{vid}: no CLIP cache entry (zeroshot needs the report split)")
        return detect_zeroshot(
            entry["t"], entry["clip_emb"], duration, verbs, verb_text, nouns, noun_text,
            encoder.logit_scale, zp,
        )  # fmt: skip

    return detect


if __name__ == "__main__":
    main()
