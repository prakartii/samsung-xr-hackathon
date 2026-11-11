"""Core data types shared by every stage (ARCHITECTURE.md §2).

All times are float seconds. Segments are (start, end) with 0 <= start < end <= T.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

Segment = tuple[float, float]
Dataset = Literal["epic_kitchens_100", "fpvsum", "unknown"]


@dataclass(frozen=True)
class VideoMeta:
    video_id: str  # filename stem; echoed into the JSON, never used for any decision
    path: Path
    duration_sec: float  # from ffprobe
    fps: float
    width: int
    height: int
    dataset: Dataset  # decided by routing.py from video content only
    route_conf: float  # signal-router margin; below cfg.routing.min_conf = uncertain


@dataclass(frozen=True, eq=False)
class Signals:
    t_dense: np.ndarray  # (N,) timestamps at cfg.signals.dense_fps
    flow: np.ndarray  # (N,) residual-motion magnitude, z-scored per video
    global_motion: np.ndarray  # (N,) camera-motion magnitude, z-scored per video
    hands: np.ndarray  # (N,) hand-presence probability in [0, 1]
    t_clip: np.ndarray  # (M,) timestamps at cfg.signals.clip_fps
    clip_emb: np.ndarray  # (M, D) L2-normalized CLIP image embeddings
    slowfast: np.ndarray | None  # (K, F) ActionFormer input features (EPIC route only)
    t_slowfast: np.ndarray | None  # (K,) timestamps of the slowfast features


@dataclass(frozen=True)
class Proposal:
    segment: Segment
    verbs: tuple[tuple[str, float], ...]  # top-k (verb_key, prob), sorted desc
    nouns: tuple[tuple[str, float], ...]  # top-k (noun_key, prob), sorted desc
    pairs: tuple[tuple[str, str, float], ...]  # top-k (verb, noun, score), sorted desc


@dataclass(frozen=True)
class Cluster:
    cluster_id: int
    span: Segment  # union span of member proposals
    member_idx: tuple[int, ...]
    pair_dist: Mapping[tuple[str, str], float]  # top-k pairs, renormalized to sum 1
    confidence: float  # max member pair score


@dataclass(frozen=True)
class VideoResult:
    team_id: str
    video_id: str
    duration_sec: float
    label: tuple[str, float]  # (name, score)
    actions: tuple[tuple[Segment, str, str, float], ...]  # (segment, verb, noun, score)
    highlights: tuple[Segment, ...]
