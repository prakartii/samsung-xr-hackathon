"""Tune the EPIC highlight path on `tune` with real ActionFormer proposals (P5/P7).

    python tools/tune_epic_selector.py [--write]

ActionFormer runs once per `tune` EPIC video (features from the cache); each config
variant then reruns exactly `rewind.analyze.summarize` (pairing → clusters → expected-
coverage selection → post-processing) and is scored with the competition coverage
(the official fraction rule, PROGRESS.md Q4, with the eval section's lambda/theta)
against the tune GT. Only `tune` is
used (CLAUDE.md §6.2).

Search: one parameter at a time from the current config (coordinate search, two
passes). A value is adopted only if `adopt` says so: mean coverage up by at least
--min-gain AND better on more videos than it is worse. With 20 videos and optimistic
proposals (ActionFormer's own training videos), small wins are noise; the rule keeps
the config unless a change is clearly better. Prints the result; --write appends a
leaderboard row for the final config. Never edits configs/default.yaml itself.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "code"), str(REPO / "tools")]

# Candidate values per parameter (current value is always included as the baseline).
GRID: dict[str, list[Any]] = {
    "cluster.tiou": [0.1, 0.2, 0.3, 0.5],
    "cluster.sim": [0.7, 0.8, 0.85, 0.9, 0.95],
    "cluster.k": [3, 5, 10],
    "select.window_lengths": [[4.05], [4.05, 5.0, 6.0], [4.05, 6.0, 8.0], [4.05, 8.0, 12.0]],
    "select.stride": [0.25, 0.5, 1.0],
    "hedge.share": [0.0, 0.1, 0.2],
    "pairing.score_power": [0.25, 0.5, 1.0],
    "pairing.alpha": [0.0, 0.5, 1.0, 2.0],
    "post.merge_gap": [0.0, 0.5, 2.0],
    "calibration.temperature": [0.5, 0.75, 1.5, 2.0, 3.0],  # S1; > 1 flattens
    "select.capture_rule": ["any_overlap", "fraction"],  # the selector's model of capture
    "select.fraction_theta": [0.4, 0.5, 0.55, 0.6, 0.7],  # used only by the fraction rule
    "select.fraction_tau": [0.02, 0.05, 0.1],
    "select.budget_frac": [0.14, 0.145],  # never above 0.145: the kit's 15% plus a margin
}


def search_keys(only: list[str] | None) -> list[str]:
    """GRID keys to search, in GRID order; --only restricts them (unknown key: error)."""
    if not only:
        return list(GRID)
    unknown = [k for k in only if k not in GRID]
    if unknown:
        raise SystemExit(f"not in GRID: {unknown}")
    return [k for k in GRID if k in only]


def with_overrides(base: dict, overrides: dict[str, Any]) -> dict:
    """Deep copy of a config dict with dotted keys set; an unknown key is a KeyError."""
    out = copy.deepcopy(base)
    for dotted, value in overrides.items():
        *path, leaf = dotted.split(".")
        node = out
        for p in path:
            node = node[p]
        if leaf not in node:
            raise KeyError(dotted)
        node[leaf] = value
    return out


def adopt(cand: dict[str, float | None], base: dict[str, float | None], min_gain: float) -> bool:
    """Mean gain ≥ min_gain over videos scored in both, and more wins than losses."""
    vids = [v for v in base if base[v] is not None and cand.get(v) is not None]
    if not vids:
        return False
    gain = sum(cand[v] - base[v] for v in vids) / len(vids)  # type: ignore[operator]
    wins = sum(cand[v] > base[v] + 1e-12 for v in vids)  # type: ignore[operator]
    losses = sum(cand[v] < base[v] - 1e-12 for v in vids)  # type: ignore[operator]
    return gain >= min_gain and wins > losses


def _mean(d: dict[str, float | None]) -> float:
    vals = [x for x in d.values() if x is not None]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    import yaml
    from import_released_features import load_fingerprint_logs

    from rewind.analyze import load_models, summarize
    from rewind.config import config_from_dict, config_hash, load_config
    from rewind.features.extract import FEATURES, signal_caches
    from rewind.features.slowfast import find_slowfast
    from rewind.utils import seed_all
    from rewind_eval.coverage import coverage
    from rewind_eval.score import LEADERBOARD, _git_sha, append_row, load_eval_data

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument("--min-gain", type=float, default=0.005)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--only", nargs="+", help="search only these GRID keys")
    args = ap.parse_args()
    keys = search_keys(args.only)

    base_dict = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg0 = load_config(args.config)
    seed_all(cfg0.run.seed)
    data = load_eval_data("tune")
    ids = sorted(data.epic_gt)
    logged = load_fingerprint_logs(sorted((REPO / "cache" / "kaggle_logs").glob("shard_*.jsonl")))
    models = load_models(cfg0, REPO)
    caches = signal_caches(models.cache_root, cfg0, models.signals)
    roots = [models.cache_root, models.released_root]
    lam, theta = cfg0.eval.coverage_lambda, cfg0.eval.capture_theta

    t0 = time.perf_counter()
    inputs = {}
    for vid in ids:  # signals + ActionFormer once per video
        fp, duration = logged[vid]
        signals = {f: caches[f].get(fp) for f in FEATURES}
        feats = find_slowfast(fp, roots)["feats"]
        dets = (models.verb_detector.detect(feats, duration),
                models.noun_detector.detect(feats, duration))  # fmt: skip
        inputs[vid] = (duration, signals, dets)
    print(f"detections for {len(ids)} videos in {time.perf_counter() - t0:.0f} s", flush=True)

    def evaluate(cfg_dict: dict) -> dict[str, float | None]:
        cfg = config_from_dict(cfg_dict)
        text, prior = models.text(cfg), models.prior(cfg)
        out = {}
        for vid, (duration, signals, dets) in inputs.items():
            r = summarize(vid, duration, signals, text, prior, cfg, lambda d=dets: d)
            out[vid] = coverage(data.epic_gt[vid], list(r.highlights), data.epic_dur[vid],
                                rule="fraction", theta=theta, lam=lam)  # fmt: skip
        return out

    best_dict, best = base_dict, evaluate(base_dict)
    print(f"current config: coverage_frac {_mean(best):.4f}", flush=True)
    changed: dict[str, Any] = {}
    for p in range(args.passes):
        moved = False
        for key in keys:
            values = GRID[key]
            current = with_overrides(best_dict, {})
            node = current
            for part in key.split("."):
                node = node[part]
            for value in values:
                if value == node:
                    continue
                cand_dict = with_overrides(best_dict, {key: value})
                cand = evaluate(cand_dict)
                ok = adopt(cand, best, args.min_gain)
                mark = "  ADOPT" if ok else ""
                print(f"pass {p + 1} {key}={value}: {_mean(cand):.4f}"
                      f" (best {_mean(best):.4f}){mark}", flush=True)  # fmt: skip
                if ok:
                    best_dict, best, changed[key], moved = cand_dict, cand, value, True
                    node = value
        if not moved:
            break
    print(f"\nfinal coverage_frac {_mean(best):.4f}; changes vs config: {changed or 'none'}")
    if args.write:
        cfg = config_from_dict(best_dict)
        append_row(LEADERBOARD, {
            "date": time.strftime("%Y-%m-%d"), "git_sha": _git_sha(),
            "tag": f"P7 tune_epic_selector {changed or 'no change'}",
            "config_hash": config_hash(cfg)[:10], "split": "tune", "n_epic": len(ids),
            "n_fpvsum": 0, "coverage_frac": f"{_mean(best):.4f}",
        })  # fmt: skip


if __name__ == "__main__":
    main()
