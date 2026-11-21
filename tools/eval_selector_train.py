"""Selector benchmark on many EPIC-100 *train* videos (decision support, not the report).

    python tools/eval_selector_train.py --n 150 \
        --variant base= --variant x1=select.confidence_power=1.0

Why: the 20-video `tune` split moves coverage by about +-0.01 between near-identical
configs (PROGRESS.md, 2026-11-21 experiments), the same size as the gains we try to
measure. EPIC-100 has 495 train videos with official annotations and ActionFormer's
released features for all of them, so selector variants can be compared on many more
videos.

What runs, per video: ActionFormer verb + noun detection on the released SlowFast
features -> `proposals_from` -> `cluster_proposals` -> `select_epic` -> `postprocess`
with the EPIC merge gap -> official coverage (>= 50% of an action inside, distinct
verb-noun pairs). Approximation: no MP4s here, so no CLIP embeddings; clusters link
proposals by matching top-1 pair only (`cluster.sim` inert) and hedges get no activity
profile. Use it for RELATIVE comparisons between selector variants.

Dev/val: ``--part dev`` (default) is the 150 videos settings are chosen on; ``--part val``
is 150 different train videos used once per experiment to confirm, never to choose. The ids
go to results/selector_splits.json.

Data rules: train videos only (ActionFormer's own training data, so detections are
optimistic, as on `tune`); never a validation/report video; the 20 `tune` videos are
left out so the two checks stay independent. Detections are cached per video under
cache/train_dets/ (gitignored). Prints mean coverage per variant and, against the first
variant, the gain and videos better/worse (the project's adopt rule).
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "code"), str(REPO / "tools")]

AF_DIR = Path(r"D:\rewind-data\actionformer\epic_kitchens")


def sample_train_ids(train_ids: list[str], exclude: set[str], n: int, seed: int) -> list[str]:
    """``n`` ids from ``train_ids`` minus ``exclude``, fixed by ``seed``, sorted."""
    pool = sorted(set(train_ids) - exclude)
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))))


def dev_val_ids(
    train_ids: list[str], exclude: set[str], n: int, seed: int
) -> tuple[list[str], list[str]]:
    """Video-level dev/val split: dev = ``n`` ids (seed), val = ``n`` more ids drawn
    (seed + 1) from what is left, so the two never share a video."""
    dev = sample_train_ids(train_ids, exclude, n, seed)
    val = sample_train_ids(train_ids, exclude | set(dev), n, seed + 1)
    return dev, val


def parse_variant(spec: str) -> tuple[str, dict[str, object]]:
    """``name=key=value;key=value`` -> (name, {key: value}); values parsed as YAML."""
    import yaml

    name, _, rest = spec.partition("=")
    overrides: dict[str, object] = {}
    for item in filter(None, rest.split(";")):
        key, _, value = item.partition("=")
        overrides[key.strip()] = yaml.safe_load(value)
    return name.strip(), overrides


def main() -> None:
    import numpy as np
    import yaml
    from tune_epic_selector import adopt, with_overrides

    from rewind.analyze import load_models, proposals_from
    from rewind.cluster import cluster_proposals
    from rewind.config import config_from_dict
    from rewind.detect.actionformer import feature_times
    from rewind.postprocess import postprocess
    from rewind.select.epic import select_epic
    from rewind.select.evidence import with_step_evidence
    from rewind.vocab import class_lists
    from rewind_eval.coverage import coverage
    from rewind_eval.gt_epic import load_epic_actions, load_video_durations
    from rewind_eval.splits import load_splits

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument("--af-dir", type=Path, default=AF_DIR)
    ap.add_argument("--variant", action="append", required=True, help="name=key=value;...")
    ap.add_argument("--min-gain", type=float, default=0.005)
    ap.add_argument(
        "--part",
        choices=["dev", "val"],
        default="dev",
        help="dev: choose settings here; val: confirm once, never tune",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "results" / "selector_splits.json",
        help="writes the dev/val ids used",
    )
    args = ap.parse_args()

    ann = REPO / "data" / "annotations" / "epic"
    classes = (ann / "EPIC_100_verb_classes.csv", ann / "EPIC_100_noun_classes.csv")
    gt = load_epic_actions(ann / "EPIC_100_train.csv", *classes)
    dur = load_video_durations(ann / "EPIC_100_video_info.csv")
    db = json.loads((args.af_dir / "annotations" / "epic_kitchens_100_verb.json").read_text())
    train = [v for v, x in db["database"].items() if x["subset"] == "training"]
    s = load_splits()
    exclude = set(s.tune_epic) | set(s.report_epic)
    dev, val = dev_val_ids(train, exclude, args.n, args.seed)
    args.manifest.write_text(json.dumps({
        "method": "EPIC-100 train videos (ActionFormer 'training' subset) minus tune and report; "
                  "dev = random.Random(seed).sample(n), val = random.Random(seed + 1).sample(n) "
                  "from the rest; video-level, disjoint",
        "seed": args.seed, "n": args.n, "dev": dev, "val": val,
    }, indent=1) + chr(10), encoding="utf-8")  # fmt: skip
    ids = [v for v in (dev if args.part == "dev" else val) if v in gt and v in dur]

    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    variants = [parse_variant(v) for v in args.variant]
    cfg0 = config_from_dict(base)
    models = load_models(cfg0, REPO)
    det_dir = REPO / "cache" / "train_dets"
    det_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    dets = {}
    for vid in ids:
        f = det_dir / f"{vid}.pkl"
        if f.exists():
            dets[vid] = pickle.loads(f.read_bytes())
            continue
        with np.load(args.af_dir / "features" / f"{vid}.npz") as z:
            feats = z["feats"]
        d = (
            models.verb_detector.detect(feats, dur[vid]),
            models.noun_detector.detect(feats, dur[vid]),
        )
        f.write_bytes(pickle.dumps(d))
        dets[vid] = d
    print(f"{len(ids)} train videos, detections in {time.perf_counter() - t0:.0f} s", flush=True)
    verbs, nouns = class_lists()
    probs = {}
    if any("step_evidence_power" in json.dumps(o) for _, o in variants):
        for vid in ids:
            with np.load(args.af_dir / "features" / f"{vid}.npz") as z:
                probs[vid] = (feature_times(len(z["v_prob"])), z["v_prob"], z["n_prob"])

    results: dict[str, dict[str, float | None]] = {}
    for name, overrides in variants:
        cfg = config_from_dict(with_overrides(base, overrides))
        prior = models.prior(cfg)
        t1 = time.perf_counter()
        cov: dict[str, float | None] = {}
        for vid in ids:
            T = dur[vid]
            props = proposals_from(dets[vid], prior, cfg)
            kw = {"span_mode": cfg.cluster.span_mode} if hasattr(cfg.cluster, "span_mode") else {}
            clusters = cluster_proposals(props, tiou_thr=cfg.cluster.tiou, sim_thr=cfg.cluster.sim,
                                         k=cfg.cluster.k, temperature=cfg.calibration.temperature,
                                         embeddings=None, **kw)  # fmt: skip
            if getattr(cfg.select, "step_evidence_power", 0.0) > 0:
                clusters = with_step_evidence(clusters, *probs[vid], verbs, nouns,
                                              cfg.select.step_evidence_power)  # fmt: skip
            segs, values = select_epic(clusters, T, cfg, mode=cfg.select.mode, profile=None)
            hl = postprocess(segs, values, T, budget_frac=cfg.select.budget_frac,
                             merge_gap=cfg.post.merge_gap_epic,
                             short_video_mode=cfg.post.short_video_mode, profile=None)  # fmt: skip
            cov[vid] = coverage(gt[vid], list(hl), T, rule="fraction", theta=0.5)
        results[name] = cov
        vals = [c for c in cov.values() if c is not None]
        line = (
            f"{name}: coverage_frac {sum(vals) / len(vals):.4f} ({time.perf_counter() - t1:.0f} s)"
        )
        if results and name != variants[0][0]:
            b = results[variants[0][0]]
            both = [v for v in ids if b[v] is not None and cov[v] is not None]
            gain = sum(cov[v] - b[v] for v in both) / len(both)  # type: ignore[operator]
            w = sum(cov[v] > b[v] + 1e-12 for v in both)  # type: ignore[operator]
            lo = sum(cov[v] < b[v] - 1e-12 for v in both)  # type: ignore[operator]
            ok = adopt(cov, b, args.min_gain)
            line += f" | vs {variants[0][0]}: {gain:+.4f}, better {w}, worse {lo}" + (
                "  ADOPT" if ok else ""
            )
        print(line, flush=True)


if __name__ == "__main__":
    main()
