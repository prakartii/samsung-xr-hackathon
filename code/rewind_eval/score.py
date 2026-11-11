"""Score predictions (or a baseline) on a split and append a leaderboard row.

    python -m rewind_eval.score --split report --pred outputs/ --tag "my run"
    python -m rewind_eval.score --split tune --baseline uniform --tag "uniform"
    python -m rewind_eval.score --split report --baseline oracle --tag "oracle"

``compute_scores`` is pure (data in, numbers out); ``main`` loads files and writes
``results/leaderboard.csv``. Missing prediction files score as empty (0 on every metric).
"""

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rewind.config import config_hash, load_config
from rewind.schema import validate
from rewind.types import Segment
from rewind.vocab import load_vocab
from rewind_eval.baselines import random_highlights, uniform_highlights
from rewind_eval.coverage import GTAction, coverage, mean_ignoring_none
from rewind_eval.f1 import frame_f1
from rewind_eval.gt_epic import load_epic_actions, load_video_durations
from rewind_eval.gt_fpvsum import FpvGT, load_fpvsum
from rewind_eval.label_acc import label_accuracy
from rewind_eval.map import mean_ap
from rewind_eval.oracle import coverage_oracle, f1_oracle
from rewind_eval.splits import load_splits

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "data" / "annotations"
LEADERBOARD = ROOT / "results" / "leaderboard.csv"
COLUMNS = [
    "date", "git_sha", "tag", "config_hash", "split", "n_epic", "n_fpvsum",
    "coverage_any", "coverage_frac", "pct_oracle", "map", "f1", "label_acc", "weighted",
]  # fmt: skip
WEIGHTS = {"coverage_any": 0.40, "map": 0.25, "f1": 0.25, "label_acc": 0.10}


@dataclass
class Prediction:
    highlights: list[Segment] = field(default_factory=list)
    actions: list[tuple[Segment, str, str, float]] = field(default_factory=list)
    label: str | None = None


@dataclass(frozen=True)
class EvalData:
    epic_gt: dict[str, list[GTAction]]
    epic_dur: dict[str, float]
    fpv_gt: dict[str, FpvGT]
    fpv_dur: dict[str, float]


def compute_scores(
    data: EvalData,
    preds: dict[str, Prediction],
    *,
    budget_frac: float,
    lam: float,
    theta: float,
    oracle_cov: dict[str, float | None],
) -> dict[str, float | None]:
    """All leaderboard metrics. ``oracle_cov``: video_id → oracle coverage (any-overlap).

    FPVSum metrics (F1, label) and the weighted total are None when the data has no
    FPVSum videos (``tune`` without ``--fold``).
    """
    empty = Prediction()
    cov_any, cov_frac = {}, {}
    for vid, gt in data.epic_gt.items():
        hl = preds.get(vid, empty).highlights
        T = data.epic_dur[vid]
        cov_any[vid] = coverage(gt, hl, T, rule="any_overlap", theta=theta, lam=lam)
        cov_frac[vid] = coverage(gt, hl, T, rule="fraction", theta=theta, lam=lam)
    c_any = mean_ignoring_none(cov_any.values())
    c_oracle = mean_ignoring_none(oracle_cov.get(v) for v in data.epic_gt)
    m = mean_ap({v: preds.get(v, empty).actions for v in data.epic_gt}, data.epic_gt)["avg"]
    f1s = [
        frame_f1(g.times, g.highlight, preds.get(v, empty).highlights)
        for v, g in data.fpv_gt.items()
    ]
    label = label_accuracy(
        {v: p.label for v, p in preds.items() if p.label is not None},
        {v: g.label for v, g in data.fpv_gt.items()},
    )
    has_fpv = bool(data.fpv_gt)
    scores: dict[str, float | None] = {
        "coverage_any": c_any,
        "coverage_frac": mean_ignoring_none(cov_frac.values()),
        "pct_oracle": c_any / c_oracle if c_oracle > 0 else 0.0,
        "map": m,
        "f1": float(np.mean(f1s)) if has_fpv else None,
        "label_acc": label if has_fpv else None,
    }
    parts = [scores[k] for k in WEIGHTS]
    scores["weighted"] = (
        None if None in parts else sum(w * scores[k] for k, w in WEIGHTS.items())  # type: ignore[operator]
    )
    return scores


def oracle_predictions(
    data: EvalData, budget_frac: float, lengths: tuple[float, ...], stride: float
) -> dict[str, Prediction]:
    """GT-derived upper reference: coverage oracle windows, GT actions, GT labels."""
    out = {}
    for vid, gt in data.epic_gt.items():
        out[vid] = Prediction(
            highlights=coverage_oracle(
                gt, data.epic_dur[vid], budget_frac=budget_frac, lengths=lengths, stride=stride
            ),
            actions=[(s, v, n, 1.0) for s, v, n in gt],
            label="cooking",
        )
    for vid, g in data.fpv_gt.items():
        hl = f1_oracle(
            g.times,
            g.highlight,
            data.fpv_dur[vid],
            budget_frac=budget_frac,
            lengths=lengths,
            stride=stride,
        )
        out[vid] = Prediction(highlights=hl, label=g.label)
    return out


def baseline_predictions(
    data: EvalData,
    kind: str,
    budget_frac: float,
    lengths: tuple[float, ...] = (4.05,),
    stride: float = 0.5,
) -> dict[str, Prediction]:
    durations = {**data.epic_dur, **data.fpv_dur}
    if kind == "uniform":
        return {
            v: Prediction(uniform_highlights(T, budget_frac), label="other")
            for v, T in durations.items()
        }
    if kind == "random":
        return {
            # Seed from the video id, so a video gets the same windows in every split.
            v: Prediction(
                random_highlights(T, budget_frac, seed=zlib.crc32(v.encode())), label="other"
            )
            for v, T in sorted(durations.items())
        }
    if kind == "oracle":
        return oracle_predictions(data, budget_frac, lengths, stride)
    raise ValueError(f"unknown baseline {kind!r}")


def read_predictions(pred_dir: Path, video_ids: list[str]) -> dict[str, Prediction]:
    """Load prediction files. A file that is missing, unreadable or fails the kit's
    validation (``strict=False``) scores as empty, i.e. 0 on every metric (kit §4)."""
    vocab = load_vocab()
    out = {}
    for vid in video_ids:
        path = pred_dir / f"{vid}.json"
        if not path.exists():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            errors, _ = validate(doc, vocab, filename_stem=vid, strict=False)
        except (OSError, ValueError) as exc:
            errors = [f"unreadable: {exc}"]
        if errors:
            print(f"WARN {path.name}: invalid, scored as empty: {errors[:2]}", file=sys.stderr)
            continue
        out[vid] = Prediction(
            highlights=[tuple(h["segment"]) for h in doc.get("highlights", [])],
            actions=[
                (tuple(a["segment"]), a["verb"], a["noun"], a["score"])
                for a in doc.get("actions", [])
            ],
            label=(doc.get("label") or {}).get("name"),
        )
    return out


def load_eval_data(split: str, fold: int | None = None) -> EvalData:
    """GT for a split. FPVSum has no separate tune videos, so ``tune`` includes FPVSum
    only for one held-out CV ``fold`` (CLAUDE.md §6.2); ``report`` has all 49."""
    s = load_splits()
    with open(ROOT / "kit" / "manifest.csv", encoding="utf-8", newline="") as f:
        manifest = {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader(f)}
    epic_dir = ANN / "epic"
    classes = (epic_dir / "EPIC_100_verb_classes.csv", epic_dir / "EPIC_100_noun_classes.csv")
    if split == "report":
        ids = list(s.report_epic)
        gt = load_epic_actions(epic_dir / "EPIC_100_validation.csv", *classes)
        epic_dur = {v: manifest[v] for v in ids}
    elif split == "tune":
        ids = list(s.tune_epic)
        gt = load_epic_actions(epic_dir / "EPIC_100_train.csv", *classes)
        info = load_video_durations(epic_dir / "EPIC_100_video_info.csv")
        epic_dur = {v: info[v] for v in ids}
    else:
        raise ValueError(f"unknown split {split!r}")
    if split == "report":
        fpv_ids = list(s.report_fpvsum)
    else:
        fpv_ids = [v for v, k in s.fpvsum_folds.items() if fold is not None and k == fold]
    fpv_dur = {v: manifest[v] for v in sorted(fpv_ids)}
    return EvalData(
        epic_gt={v: gt.get(v, []) for v in ids},
        epic_dur=epic_dur,
        fpv_gt=load_fpvsum(ANN / "fpvsum" / "gt_data.mat", fpv_dur),
        fpv_dur=fpv_dur,
    )


def _git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "code", "configs"],
            capture_output=True, text=True, check=True, cwd=ROOT,
        ).stdout.strip()  # fmt: skip
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score a run and append a leaderboard row.")
    ap.add_argument("--split", choices=["tune", "report"], required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred", type=Path, help="directory of <video_id>.json")
    src.add_argument("--baseline", choices=["uniform", "random", "oracle"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fold", type=int, help="tune only: score this FPVSum CV fold")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    budget, lam, theta = cfg.select.budget_frac, cfg.eval.coverage_lambda, cfg.eval.capture_theta
    if args.split == "report" and args.fold is not None:
        ap.error("--fold applies to --split tune only")
    data = load_eval_data(args.split, args.fold)
    if args.pred:
        preds = read_predictions(args.pred, [*data.epic_gt, *data.fpv_gt])
    else:
        preds = baseline_predictions(
            data, args.baseline, budget, cfg.select.window_lengths, cfg.select.stride
        )
    lengths, stride = cfg.select.window_lengths, cfg.select.stride
    oracle_cov = {}
    for v, gt in data.epic_gt.items():
        T = data.epic_dur[v]
        hl = coverage_oracle(gt, T, budget_frac=budget, lengths=lengths, stride=stride)
        oracle_cov[v] = coverage(gt, hl, T, rule="any_overlap", theta=theta, lam=lam)
    scores = compute_scores(
        data, preds, budget_frac=budget, lam=lam, theta=theta, oracle_cov=oracle_cov
    )
    row = {
        "date": dt.date.today().isoformat(),
        "git_sha": _git_sha(),
        "tag": args.tag,
        "config_hash": config_hash(cfg)[:10],
        "split": args.split,
        "n_epic": len(data.epic_gt),
        "n_fpvsum": len(data.fpv_gt),
        **{k: "" if v is None else f"{v:.4f}" for k, v in scores.items()},
    }
    print(json.dumps(row, indent=1))
    if not args.no_write:
        append_row(LEADERBOARD, row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
