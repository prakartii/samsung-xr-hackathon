"""Train the output-action rescorer (Campaign #4) on EPIC-100 train videos outside val.

    python tools/train_rescore.py [--write] [--val]

Rows: every output action (pairing with the output-only `actions.*` settings, before the
cap) on cached ActionFormer detections (cache/train_dets). Features:
rewind.detect.rescore.action_features (inference-time only). Label (fractional): the
mean over the mAP tIoU thresholds 0.1-0.5 of "this action is a true positive", i.e. it
takes a still-unmatched GT instance of the same (verb, noun) in the same video with tIoU
>= threshold when the video's actions are matched greedily in pairing-score order (the
order mAP itself walks). The mean matches the metric, which averages the thresholds.
Model: L2 logistic regression (tools/train_reality.fit_logistic), fit on dev + the
extra train pool (cache/train_extra_ids.json).

Model choice uses dev only: 5-fold out-of-fold (folds by video, over dev + extra)
scores give an out-of-sample dev mAP for each output transform P(TP) * old ** b.
`--val` scores the final fit on val once (confirmation, never used to choose).
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

POWERS = (0.0, 0.25, 0.5, 1.0)


def tp_labels(actions: list, gt: list, thresholds) -> np.ndarray:
    """Fraction of thresholds at which each action (already in score order) is a TP."""
    y = np.zeros(len(actions))
    for thr in thresholds:
        used: set[int] = set()
        for k, (seg, v, n, _) in enumerate(actions):
            best, best_j = -1.0, -1
            for j, (g, gv, gn) in enumerate(gt):
                if j in used or (gv, gn) != (v, n):
                    continue
                inter = max(0.0, min(seg[1], g[1]) - max(seg[0], g[0]))
                union = max(seg[1], g[1]) - min(seg[0], g[0])
                t = inter / union if union > 0 else 0.0
                if t >= thr and t > best:
                    best, best_j = t, j
            if best_j >= 0:
                used.add(best_j)
                y[k] += 1.0 / len(thresholds)
    return y


def main() -> None:
    from train_reality import auc, fit_logistic

    from rewind.analyze import load_models, proposals_from
    from rewind.config import load_config
    from rewind.constants import MAX_ACTIONS
    from rewind.detect.rescore import FEATURES, action_features
    from rewind_eval.gt_epic import load_epic_actions
    from rewind_eval.map import TIOU_THRESHOLDS, mean_ap

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--write", action="store_true", help="print the YAML block for the config")
    ap.add_argument("--val", action="store_true", help="score the final fit on val (once)")
    ap.add_argument("--drop", default="", help="comma-separated features to zero out (ablation)")
    args = ap.parse_args()

    cfg = load_config(REPO / "configs" / "default.yaml")
    a = cfg.actions
    ann = REPO / "data" / "annotations" / "epic"
    gt = load_epic_actions(ann / "EPIC_100_train.csv", ann / "EPIC_100_verb_classes.csv",
                           ann / "EPIC_100_noun_classes.csv")  # fmt: skip
    sp = json.loads((REPO / "results" / "selector_splits.json").read_text())
    extra = json.loads((REPO / "cache" / "train_extra_ids.json").read_text())
    prior = load_models(cfg, REPO).prior(cfg)
    drop = [FEATURES.index(f) for f in args.drop.split(",") if f]

    def build(ids):
        data = {}
        for vid in ids:
            f = REPO / "cache" / "train_dets" / f"{vid}.pkl"
            if vid not in gt or not f.exists():
                continue
            dets = pickle.loads(f.read_bytes())
            props = proposals_from(dets, prior, cfg, alpha=a.prior_alpha,
                                   match_tiou=a.match_tiou, keep=a.pair_keep)  # fmt: skip
            acts, x = action_features(props, dets[1], a.min_score)
            acts, x = acts[:MAX_ACTIONS], x[:MAX_ACTIONS]
            x[:, drop] = 0.0
            data[vid] = (acts, x, tp_labels(acts, gt[vid], TIOU_THRESHOLDS))
        return data

    def score_map(data, scores):
        preds = {}
        for vid, (acts, _, _) in data.items():
            s = scores[vid]
            p = [(seg, v, n, float(q)) for (seg, v, n, _), q in zip(acts, s, strict=True)]
            p.sort(key=lambda t: (-t[3], t[0][0], t[0][1], t[1], t[2]))
            preds[vid] = p[: a.max_per_video]
        return mean_ap(preds, {v: gt[v] for v in data})

    def fmt(r):
        per = " ".join(f"{k}:{r[k]:.4f}" for k in ("0.1", "0.3", "0.5"))
        return f"{r['avg']:.4f} ({per})"

    train = build(sp["dev"] + extra)
    dev_ids = [v for v in sp["dev"] if v in train]
    vids = list(train)
    xt = np.vstack([train[v][1] for v in vids])
    yt = np.concatenate([train[v][2] for v in vids])
    vt = np.concatenate([[v] * len(train[v][2]) for v in vids])
    mu, sd = xt.mean(0), xt.std(0)
    sd[sd == 0] = 1.0
    z = lambda x: (x - mu) / sd  # noqa: E731
    rng = np.random.default_rng(0)
    fold = dict(zip(rng.permutation(vids), np.arange(len(vids)) % 5, strict=True))
    oof = np.zeros(len(yt))
    for k in range(5):
        tr = np.array([fold[v] != k for v in vt])
        w = fit_logistic(z(xt[tr]), yt[tr], args.l2)
        oof[~tr] = 1.0 / (1.0 + np.exp(-(np.hstack([np.ones(((~tr).sum(), 1)), z(xt[~tr])]) @ w)))
    print(f"actions: train {len(yt)} (mean label {yt.mean():.3f}), {len(vids)} videos")
    yb = (yt > 0).astype(float)
    print(f"AUC (any-threshold TP) 5-fold: model {auc(oof, yb):.4f} vs pair score "
          f"{auc(xt[:, 0], yb):.4f}")  # fmt: skip
    dev = {v: train[v] for v in dev_ids}
    off = np.cumsum([0] + [len(train[v][2]) for v in vids])
    pos = {v: (off[i], off[i + 1]) for i, v in enumerate(vids)}
    base = score_map(dev, {v: dev[v][1][:, 0] for v in dev})
    print(f"dev mAP baseline (pair score): {fmt(base)}")
    for b in POWERS:
        sc = {v: oof[slice(*pos[v])] * dev[v][1][:, 0] ** b for v in dev}
        print(f"dev mAP OOF P * old^{b}: {fmt(score_map(dev, sc))}")

    w = fit_logistic(z(xt), yt, args.l2)
    for name, c in zip(FEATURES, w[1:], strict=True):
        print(f"  {name:16s} {c:+.3f}")
    if args.val:
        val = build(sp["val"])
        base = score_map(val, {v: val[v][1][:, 0] for v in val})
        print(f"val mAP baseline (pair score): {fmt(base)}")
        for b in POWERS:
            sc = {}
            for v in val:
                x = val[v][1]
                p = 1.0 / (1.0 + np.exp(-(np.hstack([np.ones((len(x), 1)), z(x)]) @ w)))
                sc[v] = p * x[:, 0] ** b
            print(f"val mAP P * old^{b}: {fmt(score_map(val, sc))}")
    if args.write:
        r = lambda v: "[" + ", ".join(f"{x:.6g}" for x in v) + "]"  # noqa: E731
        print(f"  rescore_coef: {r(w)}\n  rescore_mean: {r(mu)}\n  rescore_std: {r(sd)}")


if __name__ == "__main__":
    main()
