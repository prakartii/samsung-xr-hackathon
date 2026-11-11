"""Validate a directory of output JSON files against the kit rules.

    python -m rewind_eval.validate --dir outputs/            # strict (our rules)
    python -m rewind_eval.validate --dir outputs/ --kit-only # kit severities only

Checks every file with ``rewind.schema.validate`` (durations against
``kit/manifest.csv``), and across files: one JSON per manifest video, no extras, and
the same ``team_id`` everywhere. Exits 1 if there is any error.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from rewind.schema import validate
from rewind.vocab import load_vocab

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "kit" / "manifest.csv"


def read_manifest(path: Path) -> dict[str, float]:
    """video_id → duration_sec from the kit manifest."""
    with open(path, encoding="utf-8", newline="") as f:
        return {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader(f)}


def validate_dir(
    out_dir: Path, manifest: dict[str, float], *, strict: bool
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return (errors, warnings), each keyed by file name ("<dir>" for cross-file)."""
    vocab = load_vocab()
    errors: dict[str, list[str]] = {}
    warnings: dict[str, list[str]] = {}
    files = sorted(out_dir.glob("*.json"))
    team_ids: dict[str, list[str]] = {}

    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors[path.name] = [f"R1 <root>: not valid JSON ({exc})"]
            continue
        errs, warns = validate(
            doc,
            vocab,
            filename_stem=path.stem,
            expected_duration=manifest.get(path.stem),
            strict=strict,
        )
        if errs:
            errors[path.name] = errs
        if warns:
            warnings[path.name] = warns
        if isinstance(doc, dict) and isinstance(doc.get("team_id"), str):
            team_ids.setdefault(doc["team_id"], []).append(path.name)

    stems = {p.stem for p in files}
    cross_err, cross_warn = [], []
    missing = sorted(set(manifest) - stems)
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        cross_err.append(f"{len(missing)} manifest video(s) have no output: {shown}")
    extra = sorted(stems - set(manifest))
    if extra:
        cross_warn.append(f"{len(extra)} file(s) not in the manifest: {', '.join(extra[:5])}")
    if len(team_ids) > 1:
        cross_err.append(f"team_id differs across files: {sorted(team_ids)}")
    if cross_err:
        errors["<dir>"] = cross_err
    if cross_warn:
        warnings["<dir>"] = cross_warn
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate output JSONs against the kit rules.")
    parser.add_argument("--dir", type=Path, required=True, help="directory of <video_id>.json")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--kit-only", action="store_true", help="kit severities only (as the official validator)"
    )
    args = parser.parse_args(argv)

    errors, warnings = validate_dir(
        args.dir, read_manifest(args.manifest), strict=not args.kit_only
    )
    for name in sorted(warnings):
        for msg in warnings[name]:
            print(f"WARN  {name}: {msg}")
    for name in sorted(errors):
        for msg in errors[name]:
            print(f"ERROR {name}: {msg}")
    n_files = len(list(args.dir.glob("*.json")))
    n_err = sum(len(v) for v in errors.values())
    n_warn = sum(len(v) for v in warnings.values())
    print(f"{n_files} file(s): {n_err} error(s), {n_warn} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
