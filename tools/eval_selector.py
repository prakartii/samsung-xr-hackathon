"""Compare selector variants on simulated proposals (until P4's real ones exist).

    PYTHONPATH=code python tools/eval_selector.py --split tune [--write]

For each noise level (clean = GT, noisy = rewind_eval.sim.NOISY) and variant, builds
proposals from GT, clusters, selects and scores coverage (any-overlap and fraction)
and % of oracle. ``--write`` appends one leaderboard row per run (tag "P5 sim …").
Tune on ``tune`` only (CLAUDE.md §6.2).
"""

import argparse
import dataclasses
import time
from pathlib import Path

from rewind.cluster import cluster_proposals
from rewind.config import Config, config_hash, load_config
from rewind.select.epic import select_epic
from rewind_eval.baselines import uniform_highlights
from rewind_eval.coverage import coverage, mean_ignoring_none
from rewind_eval.oracle import coverage_oracle
from rewind_eval.score import LEADERBOARD, _git_sha, append_row, load_eval_data
from rewind_eval.sim import CLEAN, NOISY, simulate

ROOT = Path(__file__).resolve().parents[1]


def variants(cfg: Config) -> dict[str, tuple[Config, str]]:
    def with_(cfg, **sections):
        return dataclasses.replace(
            cfg, **{k: dataclasses.replace(getattr(cfg, k), **v) for k, v in sections.items()}
        )

    h0, h125 = {"share": 0.0}, {"share": 0.125}
    return {
        "expected h=0.125": (with_(cfg, hedge=h125), "expected"),
        "dedup h=0.125": (with_(cfg, hedge=h125), "dedup"),
        "expected h=0": (with_(cfg, hedge=h0), "expected"),
        "expected h=0 T=2": (with_(cfg, hedge=h0, calibration={"temperature": 2.0}), "expected"),
        "expected h=0 T=4": (with_(cfg, hedge=h0, calibration={"temperature": 4.0}), "expected"),
        "expected h=0 4.05": (with_(cfg, hedge=h0, select={"window_lengths": (4.05,)}), "expected"),
        "dedup h=0": (with_(cfg, hedge=h0), "dedup"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["tune", "report"], default="tune")
    ap.add_argument("--write", action="store_true", help="append leaderboard rows")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    data = load_eval_data(args.split)
    s = cfg.select
    pool = sorted({(v, n) for acts in data.epic_gt.values() for _, v, n in acts})

    def cov(hl_by_vid, rule):
        return mean_ignoring_none(
            coverage(gt, hl_by_vid[v], data.epic_dur[v], rule=rule, theta=cfg.eval.capture_theta)
            for v, gt in data.epic_gt.items()
        )

    oracle = {
        v: coverage_oracle(
            gt,
            data.epic_dur[v],
            budget_frac=s.budget_frac,
            lengths=s.window_lengths,
            stride=s.stride,
        )
        for v, gt in data.epic_gt.items()
    }
    o_any = cov(oracle, "any_overlap")
    uniform = {v: uniform_highlights(T, s.budget_frac) for v, T in data.epic_dur.items()}
    rows = [("—", "uniform", cov(uniform, "any_overlap"), cov(uniform, "fraction"), 0.0)]
    rows.append(("—", "oracle", o_any, cov(oracle, "fraction"), 0.0))

    hashes = {name: config_hash(vcfg) for name, (vcfg, _) in variants(cfg).items()}
    for noise in (CLEAN, NOISY):
        for name, (vcfg, mode) in variants(cfg).items():
            t0, hl = time.perf_counter(), {}
            for i, (v, gt) in enumerate(sorted(data.epic_gt.items())):
                T = data.epic_dur[v]
                props = simulate(gt, T, noise, pool, seed=1000 + i)
                c = vcfg.cluster
                clusters = cluster_proposals(
                    props,
                    tiou_thr=c.tiou,
                    sim_thr=c.sim,
                    k=c.k,
                    temperature=vcfg.calibration.temperature,
                )
                hl[v], _ = select_epic(clusters, T, vcfg, mode=mode)
            dt = (time.perf_counter() - t0) / len(data.epic_gt)
            rows.append((noise.name, name, cov(hl, "any_overlap"), cov(hl, "fraction"), dt))

    print(
        f"split={args.split}  n_epic={len(data.epic_gt)}  oracle (cfg window lengths) = {o_any:.4f}"
    )
    print(
        f"{'noise':6} {'variant':20} {'cov_any':>8} {'cov_frac':>8} {'%oracle':>8} {'s/video':>8}"
    )
    for noise, name, ca, cf, dt in rows:
        print(f"{noise:6} {name:20} {ca:8.4f} {cf:8.4f} {ca / o_any:8.1%} {dt:8.3f}")
        if args.write and noise != "—":
            append_row(
                LEADERBOARD,
                {
                    "date": time.strftime("%Y-%m-%d"),
                    "git_sha": _git_sha(),
                    "tag": f"P5 sim {noise}: {name}",
                    "config_hash": hashes.get(name, config_hash(cfg))[:10],
                    "split": args.split,
                    "n_epic": len(data.epic_gt),
                    "n_fpvsum": 0,
                    "coverage_any": f"{ca:.4f}",
                    "coverage_frac": f"{cf:.4f}",
                    "pct_oracle": f"{ca / o_any:.4f}",
                    "map": "",
                    "f1": "",
                    "label_acc": "",
                    "weighted": "",
                },
            )


if __name__ == "__main__":
    main()
