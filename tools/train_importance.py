"""Train the learned FPVSum importance (select/learned_importance.py) on YouTube Highlights.

    python tools/train_importance.py            # fit on every usable YTH video, print YAML

YTH (Sun et al., ECCV 2014) is the legitimate first-person highlight label set we have
(tools/eval_yth_f1.py); FPVSum's labels are never read. Label per dense step: 1 inside a
matched clip (``eval_yth_f1.positive_intervals``), else 0. Features:
``learned_importance.step_features`` on the cached flow / hands / CLIP signals plus the
CLIP similarity to ``learned_importance.prompts``. Model: L2-regularised logistic
regression (Newton, numpy), each video weighted equally with its classes balanced, so
long videos and mostly-positive videos don't dominate. Needs the cached features and the
CLIP weights (text encoder only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "code"), str(REPO / "tools")]

L2 = 10.0  # ridge on the standardised weights (fixed a priori, not tuned on a test set)


def dense_labels(t: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    """1.0 where t falls inside any [start, end) interval, else 0.0."""
    y = np.zeros(len(t))
    for a, b in intervals:
        y[(t >= a) & (t < b)] = 1.0
    return y


def video_weights(y: np.ndarray) -> np.ndarray:
    """Per-step weights: each class gets half of the video's unit weight."""
    pos = max(float(y.sum()), 1.0)
    neg = max(float((1 - y).sum()), 1.0)
    return np.where(y == 1, 0.5 / pos, 0.5 / neg)


def fit_logistic(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, iters: int = 30
) -> np.ndarray:
    """Weighted L2 logistic regression via Newton; returns [b, w...] (bias unpenalised)."""
    xb = np.hstack([np.ones((len(x), 1)), x])
    beta = np.zeros(xb.shape[1])
    reg = l2 * np.eye(xb.shape[1])
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(xb @ beta)))
        g = xb.T @ (w * (p - y)) + reg @ beta
        h = (xb * (w * p * (1 - p))[:, None]).T @ xb + reg
        step = np.linalg.solve(h, g)
        beta -= step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def main() -> None:
    from eval_yth_f1 import _list_videos, _video_meta, positive_intervals, span_matches

    from rewind.config import load_config
    from rewind.features.cache import FeatureCache, content_fingerprint
    from rewind.features.clip import ClipParams, OpenClipEncoder, clip_cache_version
    from rewind.features.flow import FlowParams, flow_cache_version
    from rewind.features.hands import HandParams, hands_cache_version
    from rewind.select.learned_importance import feature_names, step_features

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=REPO / "configs/default.yaml")
    ap.add_argument("--data-root", type=Path, default=REPO, help="holds data/ cache/ weights/")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = args.data_root

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    hp, cp = HandParams(**asdict(cfg.features.hands)), ClipParams(**asdict(cfg.features.clip))
    d = cfg.signals.dense_fps
    caches = {
        "flow": FeatureCache(root / cfg.features.cache_dir, "flow",
                             flow_cache_version(FlowParams(**asdict(cfg.features.flow)), d)),
        "hands": FeatureCache(root / cfg.features.cache_dir, "hands",
                              hands_cache_version(hp, d, sha(root / hp.model))),
        "clip": FeatureCache(root / cfg.features.cache_dir, "clip",
                             clip_cache_version(cp, cfg.signals.clip_fps, sha(root / cp.weights))),
    }  # fmt: skip
    prompts = list(cfg.learned_importance.prompts)
    text = OpenClipEncoder(cp, root).encode_texts(prompts)
    xs, ys, ws = [], [], []
    for dom, vid in _list_videos(root / "data/external/yth"):
        mp4 = root / "data/external/yth_videos" / f"{vid}.mp4"
        if not mp4.exists():
            continue
        lab = root / "data/external/yth" / dom / vid / "match_label.json"
        clips, labels = json.loads(lab.read_text())
        fps, dur = _video_meta(mp4)
        if not clips or not span_matches(clips[-1][1], fps, dur):
            continue
        gt = positive_intervals(clips, labels, fps)
        fp = content_fingerprint(mp4)
        sig = {k: c.get(fp) for k, c in caches.items()}
        if not gt or any(v is None for v in sig.values()):
            continue
        t = np.asarray(sig["flow"]["t"])
        emb = np.asarray(sig["clip"]["clip_emb"], dtype=np.float64)
        x = step_features(t, sig["flow"]["flow"], sig["hands"]["hands"],
                          sig["flow"]["global_motion"], sig["clip"]["t"], emb,
                          emb @ text.T.astype(np.float64), dur)  # fmt: skip
        y = dense_labels(t, gt)
        xs.append(x)
        ys.append(y)
        ws.append(video_weights(y))
    if not xs:
        sys.exit("no usable YTH videos with cached features")
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws) * len(xs)
    mean, std = x.mean(0), x.std(0)
    std[std == 0] = 1.0
    beta = fit_logistic((x - mean) / std, y, w, L2)
    names = feature_names(len(prompts))
    print(f"# {len(xs)} YTH videos, {len(y)} steps, {int(y.sum())} positive", file=sys.stderr)
    for n, b in zip(("bias", *names), beta, strict=True):
        print(f"#   {n:12s} {b:+.4f}", file=sys.stderr)
    fmt = lambda v: "[" + ", ".join(f"{float(a):.6g}" for a in v) + "]"  # noqa: E731
    print(f"  coef: {fmt(beta)}")
    print(f"  mean: {fmt(mean)}")
    print(f"  std: {fmt(std)}")


if __name__ == "__main__":
    main()
