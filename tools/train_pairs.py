"""Train the pair-level scorer (Campaign #3) on EPIC-100 train videos outside val.

    python tools/train_pairs.py [--write]

Rows: every (cluster, pair) the selector sees, built as tools/eval_selector_train.py builds
clusters (released features, no CLIP). Label: 1 if a GT instance of exactly that (verb,
noun) has >= 50% of its duration inside the cluster span. Features:
rewind.select.reality.pair_features (inference-time only). Model: L2 logistic regression
(tools/train_reality.fit_logistic). Fit on dev + the extra train pool
(cache/train_extra_ids.json); val is only scored (AUC vs the raw pair probability).
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


def main() -> None:
    from train_reality import auc, fit_logistic

    from rewind.analyze import load_models, proposals_from
    from rewind.cluster import cluster_proposals
    from rewind.config import load_config
    from rewind.select.reality import PAIR_FEATURES, pair_features
    from rewind_eval.gt_epic import load_epic_actions, load_video_durations

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cfg = load_config(REPO / "configs" / "default.yaml")
    ann = REPO / "data" / "annotations" / "epic"
    gt = load_epic_actions(
        ann / "EPIC_100_train.csv",
        ann / "EPIC_100_verb_classes.csv",
        ann / "EPIC_100_noun_classes.csv",
    )
    dur = load_video_durations(ann / "EPIC_100_video_info.csv")
    sp = json.loads((REPO / "results" / "selector_splits.json").read_text())
    extra = json.loads((REPO / "cache" / "train_extra_ids.json").read_text())
    prior = load_models(cfg, REPO).prior(cfg)

    def build(ids):
        xs, ys = [], []
        for vid in ids:
            f = REPO / "cache" / "train_dets" / f"{vid}.pkl"
            if vid not in gt or not f.exists():
                continue
            props = proposals_from(pickle.loads(f.read_bytes()), prior, cfg)
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
            x, keys = pair_features(cl, dur[vid])
            for i, pair in keys:
                a, b = cl[i].span
                ys.append(any((v, n) == pair and max(0.0, min(b, e) - max(a, s)) >= 0.5 * (e - s)
                              for (s, e), v, n in gt[vid]))  # fmt: skip
            xs.append(x)
        return np.vstack(xs), np.array(ys, float)

    xt, yt = build(sp["dev"] + extra)
    xv, yv = build(sp["val"])
    mu, sd = xt.mean(0), xt.std(0)
    sd[sd == 0] = 1.0
    w = fit_logistic((xt - mu) / sd, yt, args.l2)
    pv = np.hstack([np.ones((len(xv), 1)), (xv - mu) / sd]) @ w
    pi = PAIR_FEATURES.index("pair_prob")
    print(f"pairs: train {len(yt)} ({yt.mean():.1%} correct), val {len(yv)} ({yv.mean():.1%})")
    print(f"AUC val: model {auc(pv, yv):.4f} vs pair probability {auc(xv[:, pi], yv):.4f}")
    for name, c in zip(PAIR_FEATURES, w[1:], strict=True):
        print(f"  {name:14s} {c:+.3f}")
    if args.write:
        r = lambda a: "[" + ", ".join(f"{x:.6g}" for x in a) + "]"  # noqa: E731
        print(f"  pair_coef: {r(w)}\n  pair_mean: {r(mu)}\n  pair_std: {r(sd)}")


if __name__ == "__main__":
    main()
