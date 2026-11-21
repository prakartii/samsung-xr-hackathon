"""Train the cluster-reality scorer (Campaign #2) on the C2 dev train videos.

    python tools/train_reality.py   # fit on dev; AUC on dev (5-fold by video) and val

Label per cluster: 1 if its span holds >= 50% of some GT action instance (the official
capture rule, so "a window over this cluster would capture a real action"), else 0.
Features: rewind.select.reality.cluster_features (inference-time only). Model: L2-
regularised logistic regression (Newton steps, numpy). Clusters are built exactly as
tools/eval_selector_train.py builds them (released features, no CLIP).

Prints AUC of the model vs raw confidence; with --write prints the YAML block for
configs/default.yaml. Val is read only to report; nothing is fit on it.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "code"), str(REPO / "tools")]


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 30) -> np.ndarray:
    """Newton/IRLS for L2 logistic regression; returns [b, w...] on the given features."""
    xb = np.hstack([np.ones((len(x), 1)), x])
    w = np.zeros(xb.shape[1])
    reg = l2 * np.eye(xb.shape[1])
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(xb @ w)))
        g = xb.T @ (p - y) + reg @ w
        h = (xb * (p * (1 - p))[:, None]).T @ xb + reg
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def auc(score: np.ndarray, y: np.ndarray) -> float:
    order = score.argsort(kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos, neg = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)) if pos and neg else 0.5


def main() -> None:
    from rewind.analyze import load_models, proposals_from
    from rewind.cluster import cluster_proposals
    from rewind.config import load_config
    from rewind.select.reality import FEATURES, cluster_features
    from rewind_eval.gt_epic import load_epic_actions, load_video_durations

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--write", action="store_true", help="print the YAML block for the config")
    args = ap.parse_args()

    cfg = load_config(REPO / "configs" / "default.yaml")
    ann = REPO / "data" / "annotations" / "epic"
    gt = load_epic_actions(ann / "EPIC_100_train.csv", ann / "EPIC_100_verb_classes.csv",
                           ann / "EPIC_100_noun_classes.csv")  # fmt: skip
    dur = load_video_durations(ann / "EPIC_100_video_info.csv")
    splits = json.loads((REPO / "results" / "selector_splits.json").read_text())
    prior = load_models(cfg, REPO).prior(cfg)

    def build(part: str):
        xs, ys, conf, vids = [], [], [], []
        for vid in splits[part]:
            dets = pickle.loads((REPO / "cache" / "train_dets" / f"{vid}.pkl").read_bytes())
            props = proposals_from(dets, prior, cfg)
            cl = cluster_proposals(
                props,
                tiou_thr=cfg.cluster.tiou,
                sim_thr=cfg.cluster.sim,
                k=cfg.cluster.k,
                temperature=cfg.calibration.temperature,
                embeddings=None,
            )
            if not cl:
                continue
            spans = [s for s, _, _ in gt[vid]]
            for c in cl:
                a, b = c.span
                ys.append(any(max(0.0, min(b, e) - max(a, s)) >= 0.5 * (e - s) for s, e in spans))
                conf.append(c.confidence)
                vids.append(vid)
            xs.append(cluster_features(cl, dur[vid]))
        return np.vstack(xs), np.array(ys, float), np.array(conf), np.array(vids)

    xd, yd, cd, vd = build("dev")
    xv, yv, cv, _ = build("val")
    mu, sd = xd.mean(0), xd.std(0)
    sd[sd == 0] = 1.0
    z = lambda x: (x - mu) / sd  # noqa: E731
    # 5-fold cross-validation on dev, folds by video
    uv = np.unique(vd)
    rng = np.random.default_rng(0)
    fold = dict(zip(rng.permutation(uv), np.arange(len(uv)) % 5, strict=True))
    oof = np.zeros(len(yd))
    for k in range(5):
        tr = np.array([fold[v] != k for v in vd])
        w = fit_logistic(z(xd[tr]), yd[tr], args.l2)
        oof[~tr] = np.hstack([np.ones(((~tr).sum(), 1)), z(xd[~tr])]) @ w
    w = fit_logistic(z(xd), yd, args.l2)
    pv = np.hstack([np.ones((len(xv), 1)), z(xv)]) @ w
    print(f"clusters: dev {len(yd)} ({yd.mean():.1%} real), val {len(yv)} ({yv.mean():.1%} real)")
    print(f"AUC dev 5-fold: model {auc(oof, yd):.4f} vs confidence {auc(cd, yd):.4f}")
    print(f"AUC val:        model {auc(pv, yv):.4f} vs confidence {auc(cv, yv):.4f}")
    for name, coef in zip(FEATURES, w[1:], strict=True):
        print(f"  {name:12s} {coef:+.3f}")
    if args.write:
        r = lambda a: "[" + ", ".join(f"{x:.6g}" for x in a) + "]"  # noqa: E731
        print(f"  reality_coef: {r(w)}\n  reality_mean: {r(mu)}\n  reality_std: {r(sd)}")


if __name__ == "__main__":
    main()
