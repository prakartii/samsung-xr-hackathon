"""Build the verb-noun co-occurrence counts for pairing (ARCHITECTURE.md §5.1).

    python tools/build_vn_prior.py --train-csv D:/rewind-data/annotations/EPIC_100_train.csv

Reads EPIC_100_train.csv (TRAIN split only) and writes weights/vn_prior.npy: a
(98, 301) float array of counts, rows = kit verbs, columns = kit nouns, both in
kit/supported_classes.json order. Kit key i is EPIC class id i (checked at build
time against EPIC_100_{verb,noun}_classes.csv next to the train CSV; `other` is
last and gets no counts). Smoothing and normalization happen at
load time (`VNPrior.from_counts`, `pairing.prior_eps`), so the file holds raw counts.

Source: github.com/epic-kitchens/epic-kitchens-100-annotations @ ea8b404 (pinned in
PROGRESS.md P4). Leakage guard: refuses to run if any video in the CSV is one of the
evaluation videos in kit/manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def count_pairs(rows: Iterable[Mapping[str, str]], n_verbs: int, n_nouns: int) -> np.ndarray:
    """(n_verbs, n_nouns) counts of (verb_class, noun_class) over annotation rows."""
    counts = np.zeros((n_verbs, n_nouns), dtype=np.float64)
    for r in rows:
        counts[int(r["verb_class"]), int(r["noun_class"])] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "weights" / "vn_prior.npy")
    args = ap.parse_args()

    if "train" not in args.train_csv.name.lower():
        raise SystemExit(
            f"refusing {args.train_csv.name}: the prior must come from the train split"
        )
    kit = json.loads((REPO / "kit" / "supported_classes.json").read_text(encoding="utf-8"))
    verbs, nouns = kit["verbs"], kit["nouns"]
    if verbs[-1] != "other" or nouns[-1] != "other":
        raise SystemExit("kit vocabulary no longer ends with 'other': re-check the class mapping")

    with args.train_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (REPO / "kit" / "manifest.csv").open(newline="", encoding="utf-8") as f:
        eval_ids = {r["video_id"] for r in csv.DictReader(f)}
    if leaked := sorted({r["video_id"] for r in rows} & eval_ids):
        raise SystemExit(f"leakage: evaluation videos in the prior's CSV: {leaked[:5]} ...")
    # kit key i must be EPIC class i. Compare against the official class files: the
    # CSV's `verb`/`noun` columns are narrated words ("put-into"), not class keys ("insert").
    for kind, kit_keys in (("verb", verbs), ("noun", nouns)):
        path = args.train_csv.with_name(f"EPIC_100_{kind}_classes.csv")
        with path.open(newline="", encoding="utf-8") as f:
            epic = [r["key"] for r in sorted(csv.DictReader(f), key=lambda r: int(r["id"]))]
        if epic + ["other"] != kit_keys:
            raise SystemExit(f"{path.name} order differs from the kit's {kind}s + 'other'")

    counts = count_pairs(rows, len(verbs), len(nouns))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, counts)
    print(f"{len(rows)} train segments, {int((counts > 0).sum())} distinct pairs -> {args.out}")


if __name__ == "__main__":
    main()
