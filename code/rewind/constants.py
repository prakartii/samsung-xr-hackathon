"""Kit facts. These are rules, not knobs: they never go in the config.

Source for every value: the official student package, ``kit/TEAM_README.md``
(section numbers refer to its 2026-11-20 reissue).
"""

from typing import Final

#: §4 "at least one segment ≥ 4.0 s"; the first issue (2026-11-11) also said shorter
#: segments are ignored in scoring, which we keep assuming.
MIN_SEG: Final[float] = 4.0

#: Length we emit for minimum-length windows. Not a kit value: a margin above
#: MIN_SEG so that rounding to TIME_DECIMALS can never take a window below 4.0 s.
MIN_SEG_EMIT: Final[float] = 4.05

#: §4 "Target total length ≤ 15% of duration_sec" (a warning, with a soft
#: coverage penalty, §6, §9).
SCORING_BUDGET: Final[float] = 0.15

#: §4 duration_sec "must be within ±0.5 s of the official manifest.csv duration".
DURATION_TOL: Final[float] = 0.5

#: §4 segments: "0 <= start < end <= duration_sec + 0.5".
END_TOL: Final[float] = 0.5

#: §4 actions: "0–2000 items".
MAX_ACTIONS: Final[int] = 2000

#: §4 note: "Timestamps should be in seconds (up to 2 decimal places)".
TIME_DECIMALS: Final[int] = 2

#: Below this duration the single highlight we emit (MIN_SEG_EMIT) exceeds the 15%
#: budget, so R11 and R13 cannot both hold (ARCHITECTURE.md §5.6).
#: 4.05 / 0.15 = 27.0 s. (Using MIN_SEG instead, 26.67 s, would leave 26.67–27.0 s
#: videos with a 4.05 s highlight that is over budget yet not treated as short.)
SHORT_VIDEO_T: Final[float] = MIN_SEG_EMIT / SCORING_BUDGET

#: §5 "All six required fields are present"; §4 "same schema for every dataset".
REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "team_id",
    "video_id",
    "duration_sec",
    "label",
    "actions",
    "highlights",
)

#: §2 manifest.csv `dataset` values. Used offline only; inference never reads the
#: manifest (ARCHITECTURE.md §4, "Inference inputs").
DATASET_EPIC: Final[str] = "epic_kitchens_100"
DATASET_FPVSUM: Final[str] = "fpvsum"
