"""Predict FPVSum highlights + label from the feature cache, for scoring (P6).

    python tools/predict_fpvsum.py --out results/preds/fpv
    python -m rewind_eval.score --split report --pred results/preds/fpv --tag "..."
    python -m rewind_eval.score --split tune --fold 0 --pred results/preds/fpv --tag "..."

Runs the FPVSum route on each manifest FPVSum video (`data/videos/<id>.mp4`):
cached flow + hands + CLIP -> `select.importance.fpvsum_segments` -> post-processing
(ARCHITECTURE.md §5.6) -> highlights; CLIP zero-shot over the FPVSum classes
(`label.classes` without `cooking`, `label.prompts`) -> label; `actions = []` (the
kit allows it; mAP is EPIC-only). Offline evaluation tool: ids pick the files only.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import asdict, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from rewind.config import Config, load_config  # noqa: E402
from rewind.fallback import fallback_result  # noqa: E402
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
from rewind.output import dumps, to_document  # noqa: E402
from rewind.postprocess import postprocess  # noqa: E402
from rewind.select.importance import ImportanceParams, fpvsum_segments  # noqa: E402
from rewind.types import Segment  # noqa: E402
from rewind.video_io import probe_video  # noqa: E402
from rewind_eval.splits import load_splits  # noqa: E402


def fpvsum_document(
    video_id: str,
    duration: float,
    segments: list[Segment],
    label: tuple[str, float],
    cfg: Config,
) -> dict:
    """Kit document: post-processed segments as highlights, the label, no actions.

    If post-processing leaves no highlight (no segment selected), the uniform
    fallback's highlights are used so the document still satisfies kit R11.
    """
    highlights = postprocess(
        segments,
        [0.0] * len(segments),
        duration,
        budget_frac=cfg.select.budget_frac,
        merge_gap=cfg.post.merge_gap,
        short_video_mode=cfg.post.short_video_mode,
    )
    base = fallback_result(
        cfg.team.team_id,
        video_id,
        duration,
        budget_frac=cfg.select.budget_frac,
        merge_gap=cfg.post.merge_gap,
        short_video_mode=cfg.post.short_video_mode,
    )
    return to_document(replace(base, label=label, highlights=highlights or base.highlights))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--videos", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
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
    imp = ImportanceParams(**asdict(cfg.importance))
    encoder = OpenClipEncoder(clip_p, REPO)
    fpv_classes = {k: v for k, v in cfg.label.classes.items() if k != "cooking"}
    cls_emb = class_embeddings(encoder, fpv_classes, cfg.label.prompts)
    keys = list(fpv_classes)

    args.out.mkdir(parents=True, exist_ok=True)
    ids = load_splits().report_fpvsum
    for i, vid in enumerate(ids, 1):
        mp4 = args.videos / f"{vid}.mp4"
        fp = content_fingerprint(mp4)
        sig = {name: cache.get(fp) for name, cache in caches.items()}
        if any(v is None for v in sig.values()):
            raise SystemExit(f"{vid}: features not cached; run tools/cache_features.py")
        duration = probe_video(mp4).duration_sec
        segments = fpvsum_segments(
            sig["flow"], sig["hands"], sig["clip"], duration, cfg.select.budget_frac, imp
        )
        label = fpvsum_label(
            sig["clip"]["clip_emb"], cls_emb, keys, encoder.logit_scale, cfg.label.n_keyframes
        )
        doc = fpvsum_document(vid, duration, segments, label, cfg)
        (args.out / f"{vid}.json").write_text(dumps(doc), encoding="utf-8")
        print(f"[{i}/{len(ids)}] {vid}: {len(doc['highlights'])} highlights, label {label[0]}")


if __name__ == "__main__":
    main()
