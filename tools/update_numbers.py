"""Write the final `report` metrics into every document that quotes them (P9).

    python tools/update_numbers.py --weighted 0.298475 --coverage-frac 0.304458 \
        --pct-oracle 0.4999 --map 0.1492 --f1 0.2256 --label-acc 0.8298 [--dry-run]

Takes the metrics of the final leaderboard row (full precision is best; rounding
happens here) and rewrites:

- `results/ablation_report.md`, the "full pipeline" row (columns weighted,
  coverage_frac, pct_oracle, map, f1, label_acc; coverage_any only with
  --coverage-any). `tools/make_deck.py` and `tools/make_demo.py` read every number
  from this table, so rebuilding them afterwards carries the new numbers.
- `README.md`, the REWIND row of the results table; with --f1-all, the "Over all 47"
  sentence (weighted = weighted + 0.25 * (f1_all - f1)); with --tokens-pruned, the
  "% of visual tokens pruned" bullet.

Fails without writing anything if a target line is not found. The other ablation
rows and the README's "Removing a component" deltas come from `tools/ablate.py`;
rerun it (it rewrites the whole table) if they must match a new full pipeline.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ABLATION = Path("results") / "ablation_report.md"
README = Path("README.md")
FULL_ROW = "full pipeline"
# Columns of the ablation table this script may set (tools/ablate.py METRICS names).
METRICS = ("weighted", "coverage_frac", "pct_oracle", "map", "f1", "label_acc", "coverage_any")
F1_WEIGHT = 0.25  # kit weight of highlight F1 (rewind_eval.score.WEIGHTS)


class TargetNotFound(ValueError):
    """A line this script must rewrite is missing or ambiguous."""


def _one_line(lines: list[str], pred, what: str) -> int:
    hits = [i for i, ln in enumerate(lines) if pred(ln)]
    if len(hits) != 1:
        raise TargetNotFound(f"{what}: expected 1 line, found {len(hits)}")
    return hits[0]


def update_ablation(text: str, m: dict[str, float]) -> str:
    """Set the full-pipeline row's cells for the metrics in m (4 decimals)."""
    lines = text.splitlines(keepends=True)
    hdr_i = _one_line(lines, lambda ln: ln.startswith("| variant |"), "ablation header")
    header = [c.strip() for c in lines[hdr_i].strip().strip("|").split("|")]
    row_i = _one_line(
        lines, lambda ln: ln.startswith(f"| {FULL_ROW} |"), f"ablation row {FULL_ROW!r}"
    )
    cells = [c.strip() for c in lines[row_i].strip().strip("|").split("|")]
    for k, v in m.items():
        if k not in header:
            raise TargetNotFound(f"ablation column {k!r} not in header {header}")
        cells[header.index(k)] = f"{v:.4f}"
    lines[row_i] = "| " + " | ".join(cells) + " |\n"
    return "".join(lines)


def readme_row(m: dict[str, float]) -> str:
    return (
        f"| **REWIND** | **{m['weighted']:.3f}** | {m['coverage_frac']:.3f} "
        f"({100 * m['pct_oracle']:.1f}% of oracle) | {m['map']:.3f} | {m['f1']:.3f} "
        f"| {m['label_acc']:.3f} |"
    )


_ALL47 = re.compile(r"Over all 47, F1 is [0-9.]+ and the weighted total about [0-9.]+\.")
_PRUNED = re.compile(r"\*\*[0-9.]+% of visual tokens pruned\*\*")


def update_readme(
    text: str, m: dict[str, float], f1_all: float | None = None, pruned: float | None = None
) -> str:
    """REWIND results row; optionally the all-47 sentence and the tokens-pruned bullet."""
    lines = text.splitlines(keepends=True)
    i = _one_line(lines, lambda ln: ln.startswith("| **REWIND** |"), "README REWIND row")
    lines[i] = readme_row(m) + "\n"
    out = "".join(lines)
    if f1_all is not None:
        weighted_all = m["weighted"] + F1_WEIGHT * (f1_all - m["f1"])
        new = f"Over all 47, F1 is {f1_all:.3f} and the weighted total about {weighted_all:.3f}."
        out, n = _ALL47.subn(new, out)
        if n != 1:
            raise TargetNotFound(f"README 'Over all 47' sentence: found {n}")
    if pruned is not None:
        out, n = _PRUNED.subn(f"**{100 * pruned:.1f}% of visual tokens pruned**", out)
        if n != 1:
            raise TargetNotFound(f"README tokens-pruned bullet: found {n}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for k in METRICS[:-1]:
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, required=True)
    ap.add_argument("--coverage-any", type=float, help="any-overlap coverage (ablation table)")
    ap.add_argument("--f1-all", type=float, help="F1 over all 47 FPVSum videos (README note)")
    ap.add_argument("--tokens-pruned", type=float, help="fraction, e.g. 0.857 (README bullet)")
    ap.add_argument("--root", type=Path, default=REPO, help="repo root to edit")
    ap.add_argument("--dry-run", action="store_true", help="show the changes, write nothing")
    args = ap.parse_args(argv)

    m = {k: getattr(args, k) for k in METRICS if getattr(args, k) is not None}
    for k, v in m.items():
        if not 0.0 <= v <= 1.0:
            ap.error(f"--{k.replace('_', '-')} {v} is not a fraction in [0, 1]")
    edits = {
        ABLATION: lambda t: update_ablation(t, m),
        README: lambda t: update_readme(t, m, args.f1_all, args.tokens_pruned),
    }
    new: dict[Path, tuple[str, str]] = {}
    try:
        for rel, fn in edits.items():
            old = (args.root / rel).read_text(encoding="utf-8")
            new[rel] = (old, fn(old))
    except TargetNotFound as e:
        print(f"ERROR {e}; nothing written", file=sys.stderr)
        return 1
    for rel, (old, text) in new.items():
        changed = [
            (a, b) for a, b in zip(old.splitlines(), text.splitlines(), strict=True) if a != b
        ]
        print(f"{rel.as_posix()}: {len(changed)} line(s) changed")
        for a, b in changed:
            print(f"  - {a}\n  + {b}")
        if not args.dry_run:
            (args.root / rel).write_text(text, encoding="utf-8", newline="")
    if args.f1_all is None:
        print("note: README 'Over all 47' sentence left as is (pass --f1-all)")
    if args.tokens_pruned is None:
        print("note: README tokens-pruned bullet left as is (pass --tokens-pruned)")
    print("next: rebuild the deck and demo (README 'Deck and demo'); they read the ablation table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
