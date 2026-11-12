"""The model part of the pipeline: signals → route → actions, highlights, label.

ARCHITECTURE.md §4 step 4. Two layers:

- ``summarize`` is pure: cached signals, text embeddings and (optionally) ActionFormer
  detections in, a ``VideoResult`` out. Every routing rule lives here and is tested
  with small hand-made embeddings.
- ``analyze_video`` / ``load_models`` are the edge: weights, the feature cache and the
  decoder. ``pipeline.run_video`` validates whatever comes out and falls back on any
  exception.

Routes (decided from the video's content only, never the filename or manifest):
- EPIC: actions from ActionFormer when SlowFast features for this video are cached,
  else from the CLIP zero-shot detector; highlights from the expected-coverage
  selector over action clusters; label ``cooking``.
- FPVSum: no actions; highlights from the importance proxy; zero-shot label.
- Uncertain: always emit actions (a misrouted EPIC video must not lose them); the
  label decides which highlight selector runs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from rewind.cluster import cluster_proposals
from rewind.config import Config
from rewind.detect.fallback_zeroshot import (
    ZeroShotParams,
    detect_zeroshot,
    window_embeddings,
)
from rewind.detect.pairing import (
    Detection,
    PairingParams,
    VNPrior,
    pair_proposals,
    proposals_to_actions,
)
from rewind.features.cache import FeatureCache
from rewind.features.clip import Encoder, class_embeddings, zero_shot_probs
from rewind.features.extract import SignalModels, load_signals
from rewind.label import COOKING, choose_label, fpvsum_label, keyframe_indices
from rewind.postprocess import postprocess
from rewind.routing import RouteDecision, route
from rewind.select.epic import select_epic
from rewind.select.importance import (
    ImportanceParams,
    importance_signal,
    select_segments,
    shots,
)
from rewind.types import Proposal, Segment, VideoResult
from rewind.utils import get_logger
from rewind.video_io import ProbeInfo
from rewind.vocab import class_lists

log = get_logger("rewind.analyze")

Arrays = dict[str, np.ndarray]
# Bump when key_phrase or the prompt construction changes (invalidates cache/text).
TEXT_VERSION = "text-v1"
Detections = tuple[list[Detection], list[Detection]]


def key_phrase(key: str) -> str:
    """Kit key → words for a CLIP prompt: ``turn-on`` → ``turn on``,
    ``board:chopping`` → ``chopping board`` (EPIC writes the head noun first)."""
    parts = key.replace("-", " ").replace("_", " ").split(":")
    return " ".join(reversed(parts))


@dataclass(frozen=True, eq=False)
class Text:
    """Prompt-ensembled CLIP text embeddings, rows in key order."""

    logit_scale: float
    label_keys: tuple[str, ...]  # cfg.label.classes order (cooking + FPVSum classes)
    label: np.ndarray
    verb_keys: tuple[str, ...]
    verbs: np.ndarray
    noun_keys: tuple[str, ...]
    nouns: np.ndarray


def text_embeddings(encoder: Encoder, cfg: Config) -> Text:
    verbs, nouns = class_lists()
    z = cfg.zeroshot
    return Text(
        logit_scale=encoder.logit_scale,
        label_keys=tuple(cfg.label.classes),
        label=class_embeddings(encoder, cfg.label.classes, cfg.label.prompts),
        verb_keys=verbs,
        verbs=class_embeddings(encoder, {v: key_phrase(v) for v in verbs}, [z.verb_template]),
        noun_keys=nouns,
        nouns=class_embeddings(encoder, {n: key_phrase(n) for n in nouns}, [z.noun_template]),
    )


# --- pure core ----------------------------------------------------------------------


def route_video(clip_emb: np.ndarray, text: Text, cfg: Config) -> RouteDecision:
    """§6.1: per-keyframe P(cooking) from a softmax over every label class."""
    frames = np.asarray(clip_emb)[keyframe_indices(len(clip_emb), cfg.label.n_keyframes)]
    probs = zero_shot_probs(frames, text.label, text.logit_scale)
    p_cooking = probs[:, text.label_keys.index(COOKING)]
    return route(p_cooking, cfg.routing.cooking_thr, cfg.routing.min_conf)


def video_label(clip_emb: np.ndarray, text: Text, decision: RouteDecision, cfg: Config):
    """§6.3: zero-shot over the FPVSum classes (every label class but cooking)."""
    idx = [i for i, k in enumerate(text.label_keys) if k != COOKING]
    fpv = fpvsum_label(
        clip_emb,
        text.label[idx],
        [text.label_keys[i] for i in idx],
        text.logit_scale,
        cfg.label.n_keyframes,
    )
    return choose_label(decision, cfg.routing.cooking_thr, fpv, cfg.label.other_thr)


def proposals_from(dets: Detections, prior: VNPrior, cfg: Config) -> list[Proposal]:
    verbs, nouns = class_lists()
    return pair_proposals(
        dets[0], dets[1], prior, set(verbs), set(nouns), PairingParams(**asdict(cfg.pairing))
    )


def zeroshot_detections(clip: Arrays, duration: float, text: Text, cfg: Config) -> Detections:
    return detect_zeroshot(
        clip["t"],
        clip["clip_emb"],
        duration,
        text.verb_keys,
        text.verbs,
        text.noun_keys,
        text.nouns,
        text.logit_scale,
        ZeroShotParams(**asdict(cfg.zeroshot)),
    )


def epic_highlights(
    proposals: Sequence[Proposal], flow: Arrays, hands: Arrays, clip: Arrays, duration: float,
    cfg: Config,
) -> tuple[list[Segment], list[float], tuple[np.ndarray, np.ndarray]]:  # fmt: skip
    """§5.2–5.5: cluster proposals, expected-coverage windows (+ hedges)."""
    emb = (
        window_embeddings(clip["t"], clip["clip_emb"], [p.segment for p in proposals])
        if proposals
        else None
    )
    clusters = cluster_proposals(
        proposals,
        tiou_thr=cfg.cluster.tiou,
        sim_thr=cfg.cluster.sim,
        k=cfg.cluster.k,
        temperature=cfg.calibration.temperature,
        embeddings=emb,
    )
    h = cfg.hedge
    activity = h.w_flow * np.asarray(flow["flow"]) + h.w_hands * np.asarray(hands["hands"])
    profile = (np.asarray(flow["t"]), activity)
    segs, values = select_epic(clusters, duration, cfg, profile=profile)
    return segs, values, profile


def fpvsum_highlights(
    flow: Arrays, hands: Arrays, clip: Arrays, duration: float, cfg: Config
) -> tuple[list[Segment], list[float], tuple[np.ndarray, np.ndarray]]:
    """§6.2: importance proxy → best windows per shot. Value = mean importance."""
    p = ImportanceParams(**asdict(cfg.importance))
    t = np.asarray(flow["t"])
    imp = importance_signal(
        t, flow["flow"], hands["hands"], flow["global_motion"], clip["t"], clip["clip_emb"], p
    )
    spans = shots(clip["t"], clip["clip_emb"], duration, p.shot_thr)
    segs = select_segments(t, imp, spans, duration, cfg.select.budget_frac * duration, p)
    values = [float(imp[(t >= s) & (t <= e)].mean()) if np.any((t >= s) & (t <= e)) else 0.0
              for s, e in segs]  # fmt: skip
    return segs, values, (t, imp)


def summarize(
    video_id: str,
    duration: float,
    signals: dict[str, Arrays],
    text: Text,
    prior: VNPrior,
    cfg: Config,
    detect: Callable[[], Detections | None] | None = None,
) -> VideoResult:
    """Route, then actions, label and highlights for one video. Pure given ``detect``.

    detect: runs ActionFormer when this video has SlowFast features (returns None
    otherwise); called only on routes that emit actions.
    """
    flow, hands, clip = signals["flow"], signals["hands"], signals["clip"]
    if not np.array_equal(np.asarray(flow["t"]), np.asarray(hands["t"])):
        raise ValueError("flow and hands must share the dense time grid")
    decision = route_video(clip["clip_emb"], text, cfg)
    label = video_label(clip["clip_emb"], text, decision, cfg)
    log.info("route %s (margin %.3f), label %s", decision.dataset, decision.route_conf, label)

    proposals: list[Proposal] = []
    actions: list = []
    if decision.dataset != "fpvsum":  # EPIC or uncertain: always emit actions
        dets = None
        if detect is not None:
            try:
                dets = detect()
            except Exception as exc:  # never lose the route's actions to ActionFormer
                log.error("ActionFormer failed, using the zero-shot detector: %s", exc)
        if dets is None:
            dets = zeroshot_detections(clip, duration, text, cfg)
        proposals = proposals_from(dets, prior, cfg)
        actions = proposals_to_actions(proposals, cfg.actions.max_per_video, cfg.actions.min_score)

    if label[0] == COOKING:
        segs, values, profile = epic_highlights(proposals, flow, hands, clip, duration, cfg)
    else:
        segs, values, profile = fpvsum_highlights(flow, hands, clip, duration, cfg)
    highlights = postprocess(
        segs,
        values,
        duration,
        budget_frac=cfg.select.budget_frac,
        merge_gap=cfg.post.merge_gap,
        short_video_mode=cfg.post.short_video_mode,
        profile=profile,
    )
    return VideoResult(
        team_id=cfg.team.team_id,
        video_id=video_id,
        duration_sec=duration,
        label=label,
        actions=tuple(actions),
        highlights=highlights,
    )


# --- model edge ---------------------------------------------------------------------


class Detector(Protocol):
    """What ``analyze_video`` needs from ActionFormer (detect/actionformer.py)."""

    def detect(self, feats: np.ndarray, duration: float) -> list[Detection]: ...


@dataclass
class Models:
    """Everything loaded once per process. Detectors and the prior are optional."""

    signals: SignalModels
    cache_root: Path | None
    verb_detector: Detector | None = None
    noun_detector: Detector | None = None
    prior_counts: np.ndarray | None = None
    _text: dict = field(default_factory=dict)

    def text(self, cfg: Config) -> Text:
        """Text embeddings for this config: memory, then the cache, then CLIP.

        run.sh is one process per video and encoding the ~450 prompts takes ~30 s on
        CPU, so they are cached on disk too (cache version = CLIP version; entry key =
        hash of every prompt input).
        """
        verbs, nouns = class_lists()
        # No sort_keys: the label.classes order is the row order of the cached arrays.
        inputs = [TEXT_VERSION, asdict(cfg.label), asdict(cfg.zeroshot), verbs, nouns]
        key = hashlib.sha1(json.dumps(inputs).encode()).hexdigest()
        if key in self._text:
            return self._text[key]
        cache = (
            FeatureCache(self.cache_root, "text", self.signals.clip_version)
            if self.cache_root is not None
            else None
        )
        hit = cache.get(key) if cache is not None else None
        if hit is not None and tuple(hit["label_keys"].tolist()) != tuple(cfg.label.classes):
            hit = None  # can't happen with the ordered key; never trust misaligned rows
        if hit is not None:
            text = Text(float(hit["logit_scale"]), tuple(cfg.label.classes), hit["label"],
                        verbs, hit["verbs"], nouns, hit["nouns"])  # fmt: skip
        else:
            text = text_embeddings(self.signals.clip, cfg)
            if cache is not None:
                arrays = {"label": text.label, "verbs": text.verbs, "nouns": text.nouns}
                arrays["label_keys"] = np.asarray(text.label_keys)
                try:
                    cache.put(key, {**arrays, "logit_scale": np.asarray(text.logit_scale)})
                except OSError as exc:  # read-only mount, full disk
                    log.warning("text embeddings not cached: %s", exc)
        self._text[key] = text
        return text

    def prior(self, cfg: Config) -> VNPrior:
        verbs, nouns = class_lists()
        counts = self.prior_counts
        if counts is None:  # no prior file: uniform (every pair equally likely)
            counts = np.zeros((len(verbs), len(nouns)))
        return VNPrior.from_counts(verbs, nouns, counts, cfg.pairing.eps)


def load_models(cfg: Config, repo_root: Path) -> Models:
    """Real models. Raises FileNotFoundError before importing torch if CLIP or the
    hand model is missing (then ``run_video`` writes the fallback)."""
    from rewind.features.cache import sha256_file
    from rewind.features.clip import OpenClipEncoder, clip_cache_version
    from rewind.features.extract import ExtractParams
    from rewind.features.hands import MediaPipeHands, hands_cache_version

    p = ExtractParams.from_config(cfg)
    clip_w, hand_w = repo_root / p.clip.weights, repo_root / p.hands.model
    for w in (clip_w, hand_w):
        if not w.exists():
            raise FileNotFoundError(f"{w} missing: run tools/fetch_weights.sh")
    signals = SignalModels(
        clip=OpenClipEncoder(p.clip, repo_root),
        new_hands=lambda: MediaPipeHands(p.hands, repo_root),
        clip_version=clip_cache_version(p.clip, p.clip_fps, sha256_file(clip_w)),
        hands_version=hands_cache_version(p.hands, p.dense_fps, sha256_file(hand_w)),
    )
    models = Models(signals=signals, cache_root=repo_root / cfg.features.cache_dir)

    d = cfg.detect
    prior = repo_root / d.vn_prior
    verbs, nouns = class_lists()
    if prior.exists():
        counts = np.load(prior, allow_pickle=False)
        if counts.shape == (len(verbs), len(nouns)):
            models.prior_counts = counts
        else:
            log.error("%s has shape %s, not %s: uniform prior", prior, counts.shape,
                      (len(verbs), len(nouns)))  # fmt: skip
    else:
        log.warning("%s missing: pairing uses a uniform verb-noun prior", prior)
    ckpts = [repo_root / d.verb_checkpoint, repo_root / d.noun_checkpoint]
    if all(c.exists() for c in ckpts):
        from rewind.detect.actionformer import ActionFormerDetector, ActionFormerParams

        try:
            verb = ActionFormerDetector(
                ActionFormerParams(d.verb_config, d.verb_checkpoint), verbs, repo_root
            )
            noun = ActionFormerDetector(
                ActionFormerParams(d.noun_config, d.noun_checkpoint), nouns, repo_root
            )
            models.verb_detector, models.noun_detector = verb, noun
        except Exception as exc:  # e.g. a checkpoint the installed torch can't load
            log.error("ActionFormer failed to load, EPIC actions use zero-shot: %s", exc)
    else:
        log.warning("ActionFormer checkpoints missing: EPIC actions use the zero-shot detector")
    return models


def analyze_video(
    video: Path, probe: ProbeInfo, video_id: str, cfg: Config, models: Models
) -> VideoResult:
    """Signals (cache or decode) → ``summarize``. ``video_id`` is only echoed."""
    duration = probe.duration_sec
    signals = load_signals(video, duration, cfg, models.signals, models.cache_root)

    def detect() -> Detections | None:
        if models.verb_detector is None or models.noun_detector is None:
            return None
        if models.cache_root is None:
            return None
        from rewind.features.cache import content_fingerprint
        from rewind.features.slowfast import slowfast_cache

        entry = slowfast_cache(models.cache_root).get(content_fingerprint(video))
        if entry is None:
            log.info("no SlowFast features cached for this video: zero-shot detector")
            return None
        feats = entry["feats"]
        return (
            models.verb_detector.detect(feats, duration),
            models.noun_detector.detect(feats, duration),
        )

    return summarize(video_id, duration, signals, models.text(cfg), models.prior(cfg), cfg, detect)
