"""Poll open PRs and print the ones this side's review agent must act on.

Used by the review sub-agent described in CLAUDE.md §5 ("Review agents").

    python tools/pr_watch.py                  # list new items (does not mark them)
    python tools/pr_watch.py --done <key>     # mark one item as handled
    python tools/pr_watch.py --me <login>     # override the GitHub login (default: gh user)

Each output line is tab-separated: key, kind, #number, author, checks, title.

- kind ``review``: an open, non-draft PR by someone else whose head commit we haven't
  reviewed yet. New commits produce a new key, so the PR is reviewed again.
- kind ``own``: one of our open PRs that has reviews or comments, or failing checks, in a
  state we haven't reported yet.

Handled keys are stored in ``<git common dir>/rewind-pr-watch.seen`` (local, never
committed; shared by all worktrees of this clone).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PR_FIELDS = "number,headRefOid,author,title,isDraft,reviews,comments,statusCheckRollup"
_FAILED = {"FAILURE", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED", "STARTUP_FAILURE", "ERROR"}
_OK = {"SUCCESS", "NEUTRAL", "SKIPPED"}


def check_state(rollup: list[dict[str, Any]]) -> str:
    """Summarize a PR's statusCheckRollup as none | failing | pending | passing."""
    if not rollup:
        return "none"
    states = []
    for c in rollup:
        if c.get("__typename") == "StatusContext":
            states.append(c.get("state") or "PENDING")
        elif c.get("status") != "COMPLETED":
            states.append("PENDING")
        else:
            states.append(c.get("conclusion") or "PENDING")
    if any(s in _FAILED for s in states):
        return "failing"
    if all(s in _OK for s in states):
        return "passing"
    return "pending"


def pending_items(prs: list[dict[str, Any]], me: str, seen: set[str]) -> list[dict[str, str]]:
    """Items needing attention, oldest PR first. Pure: no I/O."""
    items = []
    for pr in sorted(prs, key=lambda p: p["number"]):
        author = pr["author"]["login"]
        sha = pr["headRefOid"]
        checks = check_state(pr.get("statusCheckRollup") or [])
        if author != me:
            if pr.get("isDraft"):
                continue
            key = f"review:{pr['number']}:{sha}"
            kind = "review"
        else:
            feedback = len(pr.get("reviews") or []) + len(pr.get("comments") or [])
            if feedback == 0 and checks != "failing":
                continue
            key = f"own:{pr['number']}:{sha}:{feedback}:{checks}"
            kind = "own"
        if key in seen:
            continue
        items.append(
            {
                "key": key,
                "kind": kind,
                "number": str(pr["number"]),
                "author": author,
                "checks": checks,
                "title": pr["title"],
            }
        )
    return items


def _git_common_dir() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"], capture_output=True, text=True, check=True
    )
    return Path(out.stdout.strip()).resolve()


def _seen_file() -> Path:
    return _git_common_dir() / "rewind-pr-watch.seen"


def _read_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _gh_json(args: list[str]) -> Any:
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--me", help="our GitHub login (default: the gh user)")
    parser.add_argument("--done", metavar="KEY", help="mark KEY as handled and exit")
    args = parser.parse_args(argv)

    seen_path = _seen_file()
    if args.done:
        with seen_path.open("a", encoding="utf-8") as f:
            f.write(args.done + "\n")
        return 0

    me = args.me or _gh_json(["api", "user"])["login"]
    prs = _gh_json(["pr", "list", "--state", "open", "--limit", "100", "--json", PR_FIELDS])
    for item in pending_items(prs, me, _read_seen(seen_path)):
        print(
            "\t".join(
                [
                    item["key"],
                    item["kind"],
                    f"#{item['number']}",
                    item["author"],
                    item["checks"],
                    item["title"],
                ]
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
