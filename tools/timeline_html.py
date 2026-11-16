"""Timeline viewer: one static HTML page over our output JSONs and their videos (S4).

    python tools/timeline_html.py --json outputs/P01_11.json outputs/Bike_lhTFVa0h9Rk.json \
        --videos data/videos --out demo/timeline.html

For the demo only (REWIND_v2.3_plan.md §10: ffmpeg reel + one static HTML page). The
page plays the original MP4 with our highlights and detected actions drawn on a
timeline under it: click anywhere to seek, click a highlight or action to jump to it,
and "Highlights only" plays just the selected windows back to back. Everything is
inlined (no libraries, no network); videos are referenced by a path relative to the
page, so open it from disk next to data/videos. Missing videos still show the timeline.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def page_entry(doc: dict, video_src: str | None, max_actions: int) -> dict:
    """What the page needs from one output JSON: highlights in time order and the
    ``max_actions`` highest-scoring actions (ties by start time), rounded."""
    actions = sorted(doc.get("actions", []), key=lambda a: (-a["score"], a["segment"][0]))
    return {
        "video_id": doc["video_id"],
        "src": video_src,
        "duration": float(doc["duration_sec"]),
        "label": doc["label"]["name"],
        "label_conf": round(float(doc["label"]["score"]), 3),
        "highlights": sorted(
            [round(float(s), 2), round(float(e), 2)]
            for s, e in (h["segment"] for h in doc["highlights"])
        ),
        "actions": [
            {
                "v": a["verb"],
                "n": a["noun"],
                "s": round(float(a["segment"][0]), 2),
                "e": round(float(a["segment"][1]), 2),
                "p": round(float(a["score"]), 3),
            }
            for a in actions[:max_actions]
        ],
        "n_actions": len(doc.get("actions", [])),
    }


def video_src(video_dir: Path, video_id: str, out: Path) -> str | None:
    """Path of <video_dir>/<video_id>.mp4 relative to the page (URL slashes), a file://
    URI when it is on another drive, or None when the video is missing."""
    mp4 = video_dir / f"{video_id}.mp4"
    if not mp4.exists():
        return None
    try:
        return Path(os.path.relpath(mp4.resolve(), out.resolve().parent)).as_posix()
    except ValueError:  # different drive (Windows): no relative path exists
        return mp4.resolve().as_uri()


def script_json(data: object) -> str:
    """JSON safe inside <script>: no '</' or '<!--' can close or confuse the tag."""
    return json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")


def render(entries: list[dict]) -> str:
    title = "REWIND timeline"
    template = (Path(__file__).parent / "timeline_template.html").read_text(encoding="utf-8")
    return template.replace("__TITLE__", html.escape(title)).replace(
        "__DATA__", script_json(entries)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--json", type=Path, nargs="+", required=True, help="output JSON file(s)")
    ap.add_argument("--videos", type=Path, default=Path("data/videos"))
    ap.add_argument("--out", type=Path, default=Path("demo/timeline.html"))
    ap.add_argument("--max-actions", type=int, default=300, help="top actions drawn per video")
    args = ap.parse_args()

    entries = []
    for p in args.json:
        doc = json.loads(p.read_text(encoding="utf-8"))
        src = video_src(args.videos, doc["video_id"], args.out)
        entries.append(page_entry(doc, src, args.max_actions))
        found = "found" if src else "MISSING"
        print(
            f"{doc['video_id']}: {len(doc['highlights'])} highlights, "
            f"{len(doc.get('actions', []))} actions, video {found}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(entries), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
