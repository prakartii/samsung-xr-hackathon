"""FPVSum ground truth → per-frame importance and highlight labels per manifest video.

``gt_data.mat`` holds, per annotated video: ``Score`` (mean annotator importance per
native frame, 0–1) and ``Binary_score`` (the dataset's own top-15% highlight frames).
Frame i is placed at ``i · T / N`` seconds, with T the manifest duration.

Names: the manifest uses ``<Category>_<YouTubeID>``; gt_data strips leading
underscores from the ID (``Jumping__mU-p2wuk90`` → ``Jumping_mU-p2wuk90``), so names
are matched on (category, ID without leading underscores).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, eq=False)
class FpvGT:
    times: np.ndarray  # (N,) seconds, frame starts
    score: np.ndarray  # (N,) mean importance in [0, 1]
    highlight: np.ndarray  # (N,) bool, the dataset's top-15% frames
    label: str  # kit label, e.g. "rockclimbing"


def name_key(name: str) -> tuple[str, str]:
    category, _, yt = name.partition("_")
    return category.lower(), yt.lstrip("_")


def load_fpvsum(gt_mat: Path, durations: dict[str, float]) -> dict[str, FpvGT]:
    """GT for every video in ``durations`` (video_id → manifest seconds) that has one."""
    import scipy.io  # eval-only dependency

    records = scipy.io.loadmat(gt_mat, squeeze_me=True, struct_as_record=False)["gt_data"]
    by_key = {name_key(str(r.Name)): r for r in np.atleast_1d(records)}
    out: dict[str, FpvGT] = {}
    for vid, duration in durations.items():
        r = by_key.get(name_key(vid))
        if r is None:
            continue
        score = np.asarray(r.Score, dtype=float).ravel()
        n = len(score)
        out[vid] = FpvGT(
            times=np.arange(n, dtype=float) * duration / n,
            score=score,
            highlight=np.asarray(r.Binary_score).ravel().astype(bool),
            label=name_key(vid)[0],
        )
    return out
