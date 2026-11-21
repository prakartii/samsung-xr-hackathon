"""Output-action mAP on the C2 dev/val EPIC-100 train splits (decision support).

    python tools/eval_actions_train.py --part dev --variant base= --variant k25=actions.pair_keep=25

Runs `rewind.analyze.output_actions` (pairing with the output-only `actions.*` settings,
per-class soft-NMS, cap) on cached ActionFormer detections (cache/train_dets, written by
tools/eval_selector_train.py) for the ids in results/selector_splits.json, and scores
ActivityNet-style action mAP (tIoU 0.1-0.5) against the official train annotations.

Caveat: train videos are ActionFormer's own training data, so absolute mAP is inflated
(dev ~0.51 vs 0.149 on the scored val set). Use it only to compare output-pairing
variants, decided on dev and confirmed once on val. Never a report video, never FPVSum.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "code"), str(REPO / "tools")]


def duplicate_rate(actions: list, thr: float = 0.5) -> float:
    """Share of actions that overlap (tIoU >= thr) a higher-scored action of the same class."""
    from rewind_eval.map import tiou

    seen: dict[tuple[str, str], list] = {}
    dup = 0
    for seg, v, n, _ in sorted(actions, key=lambda a: -a[3]):
        prev = seen.setdefault((v, n), [])
        dup += any(tiou(seg, p) >= thr for p in prev)
        prev.append(seg)
    return dup / len(actions) if actions else 0.0


def main() -> None:
    import yaml
    from eval_selector_train import parse_variant
    from tune_epic_selector import with_overrides

    from rewind.analyze import load_models, output_actions
    from rewind.config import config_from_dict
    from rewind_eval.gt_epic import load_epic_actions
    from rewind_eval.map import mean_ap

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--part", choices=["dev", "val"], default="dev")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument("--splits", type=Path, default=REPO / "results" / "selector_splits.json")
    ap.add_argument("--variant", action="append", required=True, help="name=key=value;...")
    args = ap.parse_args()

    ann = REPO / "data" / "annotations" / "epic"
    gt = load_epic_actions(ann / "EPIC_100_train.csv", ann / "EPIC_100_verb_classes.csv",
                           ann / "EPIC_100_noun_classes.csv")  # fmt: skip
    ids = [v for v in json.loads(args.splits.read_text())[args.part] if v in gt]
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = load_models(config_from_dict(base), REPO)
    dets = {v: pickle.loads((REPO / "cache" / "train_dets" / f"{v}.pkl").read_bytes()) for v in ids}
    gts = {v: gt[v] for v in ids}
    for name, overrides in (parse_variant(s) for s in args.variant):
        cfg = config_from_dict(with_overrides(base, overrides))
        prior = models.prior(cfg)
        t0 = time.perf_counter()
        preds = {v: output_actions(dets[v], prior, cfg) for v in ids}
        dt = time.perf_counter() - t0
        res = mean_ap(preds, gts)
        n = sum(len(p) for p in preds.values()) / len(ids)
        dup = sum(duplicate_rate(p) for p in preds.values()) / len(ids)
        per = " ".join(f"{k}:{res[k]:.4f}" for k in ("0.1", "0.3", "0.5"))
        print(f"{name}: mAP {res['avg']:.4f} ({per}) | actions/video {n:.0f} | "
              f"dup rate {dup:.3f} | pairing {dt:.1f} s", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
