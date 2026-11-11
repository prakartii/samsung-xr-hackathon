"""EPIC-KITCHENS-100 ground truth → GT actions per video, in kit class keys.

The kit's verbs/nouns are EPIC-100's class keys in id order plus "other", so a row's
``verb_class``/``noun_class`` id maps to ``EPIC_100_{verb,noun}_classes.csv`` → key.
The raw ``verb``/``noun`` columns (e.g. "put-down") are NOT class keys and are unused.
"""

import csv
from collections import defaultdict
from pathlib import Path

from rewind.types import Segment

GTAction = tuple[Segment, str, str]


def parse_timestamp(ts: str) -> float:
    """'HH:MM:SS.ff' → seconds."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def read_class_keys(path: Path) -> dict[int, str]:
    with open(path, encoding="utf-8", newline="") as f:
        return {int(r["id"]): r["key"] for r in csv.DictReader(f)}


def load_epic_actions(
    annotations_csv: Path, verb_classes_csv: Path, noun_classes_csv: Path
) -> dict[str, list[GTAction]]:
    """video_id → [(segment, verb_key, noun_key)], sorted by start then end."""
    verbs = read_class_keys(verb_classes_csv)
    nouns = read_class_keys(noun_classes_csv)
    out: dict[str, list[GTAction]] = defaultdict(list)
    with open(annotations_csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            s, e = parse_timestamp(r["start_timestamp"]), parse_timestamp(r["stop_timestamp"])
            if not e > s:
                continue  # a handful of zero-length narrations can't be matched
            out[r["video_id"]].append(
                ((s, e), verbs[int(r["verb_class"])], nouns[int(r["noun_class"])])
            )
    return {vid: sorted(acts) for vid, acts in out.items()}


def load_video_durations(video_info_csv: Path) -> dict[str, float]:
    with open(video_info_csv, encoding="utf-8", newline="") as f:
        return {r["video_id"]: float(r["duration"]) for r in csv.DictReader(f)}
