"""The tune / report video lists (configs/splits.yaml, made by tools/make_splits.py)."""

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_SPLITS = Path(__file__).resolve().parents[2] / "configs" / "splits.yaml"


@dataclass(frozen=True)
class Splits:
    tune_epic: tuple[str, ...]  # EPIC-100 train videos, never val
    fpvsum_folds: dict[str, int]  # video_id → CV fold
    report_epic: tuple[str, ...]  # the 138 manifest EPIC val videos
    report_fpvsum: tuple[str, ...]  # the 49 manifest FPVSum videos (all predicted)
    unscored_fpvsum: frozenset[str] = frozenset()  # of those, left out of official scoring


def load_splits(path: Path = DEFAULT_SPLITS) -> Splits:
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return Splits(
        tune_epic=tuple(d["tune"]["epic"]),
        fpvsum_folds=dict(d["tune"]["fpvsum_folds"]),
        report_epic=tuple(d["report"]["epic"]),
        report_fpvsum=tuple(d["report"]["fpvsum"]),
        unscored_fpvsum=frozenset(d["report"].get("unscored_fpvsum", [])),
    )
