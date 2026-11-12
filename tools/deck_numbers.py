"""The deck's three numbers (P8; REWIND_v2.3_plan.md §15), reproducibly.

    python tools/deck_numbers.py --pred results/preds/full_report --tag "<leaderboard tag>"

1. % of oracle coverage: `pct_oracle` of the leaderboard row with --tag (EPIC, report).
2. Visual tokens pruned: 1 - (highlight seconds / video seconds) pooled over every
   output in --pred (frames kept vs total at any fixed frame rate).
3. CPU seconds per minute of video: pooled over results/latency.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def tokens_pruned(docs: list[dict]) -> float:
    kept = sum(e - s for d in docs for s, e in (h["segment"] for h in d["highlights"]))
    return 1.0 - kept / sum(float(d["duration_sec"]) for d in docs)


def sec_per_video_minute(rows: list[dict]) -> float:
    return sum(float(r["seconds"]) for r in rows) / (
        sum(float(r["duration_sec"]) for r in rows) / 60.0
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="leaderboard tag of the report run")
    args = ap.parse_args()

    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.pred.glob("*.json"))]
    with (REPO / "results" / "leaderboard.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["tag"] == args.tag]
    if not rows:
        raise SystemExit(f"no leaderboard row tagged {args.tag!r}")
    with (REPO / "results" / "latency.csv").open(newline="", encoding="utf-8") as f:
        lat = list(csv.DictReader(f))
    print(f"1. {100 * float(rows[-1]['pct_oracle']):.1f}% of oracle coverage (EPIC report)")
    print(f"2. {100 * tokens_pruned(docs):.1f}% of visual tokens pruned ({len(docs)} videos)")
    print(f"3. {sec_per_video_minute(lat):.1f} s of CPU per minute of video ({len(lat)} timed)")


if __name__ == "__main__":
    main()
