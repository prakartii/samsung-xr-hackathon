"""Kit facts. These are rules, not knobs: they never go in the config.

Source for every value: the official student package, ``kit/TEAM_README.md``
(section numbers refer to that file).
"""

from typing import Final

#: §3 "at least one segment >= 4.0 s"; §4 shorter segments are ignored in scoring.
MIN_SEG: Final[float] = 4.0

#: Length we emit for minimum-length windows. Not a kit value: a margin above
#: MIN_SEG so that rounding to TIME_DECIMALS can never take a window below 4.0 s.
MIN_SEG_EMIT: Final[float] = 4.05

#: §3 "Target total length <= 15% of duration_sec" (a warning, with a soft
#: coverage penalty, §4–§5).
SCORING_BUDGET: Final[float] = 0.15

#: §3 duration_sec "must be within ±0.5 s of the official manifest.csv duration".
DURATION_TOL: Final[float] = 0.5

#: §3 segments: "0 <= start < end <= duration_sec + 0.5".
END_TOL: Final[float] = 0.5

#: §3 actions: "0–2000 items".
MAX_ACTIONS: Final[int] = 2000

#: §3 note: "Timestamps should be in seconds (up to 2 decimal places)".
TIME_DECIMALS: Final[int] = 2

#: Below this duration a single MIN_SEG highlight exceeds the budget, so R11 and
#: R13 cannot both hold (ARCHITECTURE.md §5.6). 4.0 / 0.15 = 26.67 s.
SHORT_VIDEO_T: Final[float] = MIN_SEG / SCORING_BUDGET

#: §3 "Both datasets use the exact same 6 top-level fields".
REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "team_id",
    "video_id",
    "duration_sec",
    "label",
    "actions",
    "highlights",
)

#: §1 manifest.csv `dataset` values. Used offline only; inference never reads the
#: manifest (ARCHITECTURE.md §4, "Inference inputs").
DATASET_EPIC: Final[str] = "epic_kitchens_100"
DATASET_FPVSUM: Final[str] = "fpvsum"
