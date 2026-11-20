"""Ablation table: the full pipeline vs one component removed at a time (P7, deck).

    python tools/ablate.py --split report [--write]
    python tools/ablate.py --split tune   [--write]

Each variant is the current config with one change (or ActionFormer switched off),
run through `predict_cached.Predictor` (the exact `analyze.summarize` path, from the
feature cache) and scored with `rewind_eval.score`; uniform and oracle baselines are
added. `tune` combines its 20 EPIC videos with the mean of the 5 FPVSum CV folds.

Measuring only: nothing here may choose a setting from `report` (CLAUDE.md §6.2);
decisions such as the P5 kill rule (expected coverage vs label dedup) use `--split
tune`. Writes results/ablation_<split>.md; --write also appends leaderboard rows.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "code"), str(REPO / "tools")]

# Official coverage first (fraction rule, Q4); any-overlap kept last for reference.
METRICS = ["weighted", "coverage_frac", "pct_oracle", "map", "f1", "label_acc", "coverage_any"]
N_FOLDS = 5

# name -> (dotted config overrides, ActionFormer on)
VARIANTS: dict[str, tuple[dict, bool]] = {
    "full pipeline": ({}, True),
    "zero-shot actions (no ActionFormer)": ({}, False),
    "label-dedup selector (no expected coverage)": ({"select.mode": "dedup"}, True),
    "no verb-noun prior (selector too)": ({"pairing.alpha": 0.0}, True),
    "no hand signal": ({"importance.w_hands": 0.0, "hedge.w_hands": 0.0}, True),
    "no flow / camera motion": (
        {"importance.w_flow": 0.0, "importance.w_global_motion": 0.0, "hedge.w_flow": 0.0}, True),
    "no scene change (FPVSum)": ({"importance.w_scene_change": 0.0}, True),
}  # fmt: skip


def combine_tune(epic: dict, folds: list[dict]) -> dict[str, float]:
    """tune metrics: EPIC from its row, FPVSum as the mean over the fold rows, and
    the weighted total recomputed from those (kit weights)."""
    out = {k: float(epic[k]) for k in ("coverage_any", "coverage_frac", "pct_oracle", "map")}
    for k in ("f1", "label_acc"):
        out[k] = sum(float(r[k]) for r in folds) / len(folds)
    from rewind_eval.score import WEIGHTS  # one source for the kit weights

    out["weighted"] = sum(w * out[k] for k, w in WEIGHTS.items())
    return out


def _score(argv: list[str]) -> dict:
    from rewind_eval.score import main as score_main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        score_main(argv)
    text = buf.getvalue()
    return json.loads(text[text.index("{") :])


def score_variant(split: str, src: list[str], tag: str, write: bool) -> dict[str, float]:
    base = ["--split", split, *src, "--tag", tag] + ([] if write else ["--no-write"])
    if split == "report":
        row = _score(base)
        return {k: float(row[k]) for k in METRICS if row[k] != ""}
    folds = [_score([*base, "--fold", str(k)]) for k in range(N_FOLDS)]
    return combine_tune(_score(base), folds)


def main() -> None:
    import yaml
    from predict_cached import Predictor, default_logs, split_ids
    from tune_epic_selector import with_overrides

    from rewind.config import config_from_dict, config_hash

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=["tune", "report"], required=True)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument("--videos", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--write", action="store_true", help="append leaderboard rows")
    args = ap.parse_args()

    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg0 = config_from_dict(base)
    predictor = Predictor(cfg0, args.videos, default_logs())
    ids = split_ids(args.split)
    rows: dict[str, dict[str, float]] = {}
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        for name, (overrides, af) in VARIANTS.items():
            cfg = config_from_dict(with_overrides(base, overrides))
            out = Path(tmp) / f"v{len(rows)}"
            with contextlib.redirect_stdout(io.StringIO()):
                predictor.run(cfg, ids, out, actionformer=af)
            rows[name] = score_variant(args.split, ["--pred", str(out)], f"ablate: {name}",
                                       args.write)  # fmt: skip
            print(f"{name}: weighted {rows[name].get('weighted', float('nan')):.4f}"
                  f"  ({time.perf_counter() - t0:.0f} s)", flush=True)  # fmt: skip
    for b in ("uniform", "oracle"):
        rows[f"{b} baseline"] = score_variant(args.split, ["--baseline", b], f"ablate: {b}", False)

    lines = [f"Ablation on `{args.split}` (tools/ablate.py, config "
             f"{config_hash(cfg0)[:10]}).", "",
             "| variant | " + " | ".join(METRICS) + " |",
             "|---|" + "---|" * len(METRICS)]  # fmt: skip
    for name, r in rows.items():
        cells = [f"{r[m]:.4f}" if m in r else "" for m in METRICS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    (REPO / "results" / f"ablation_{args.split}.md").write_text(table, encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()
