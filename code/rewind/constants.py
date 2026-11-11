"""Kit facts: rules from the official student kit, not tunable knobs (CLAUDE.md §6.3).

Values exactly as listed in ARCHITECTURE.md §3.1; each cites kit/TEAM_README.md.
Tunable numbers live in configs/default.yaml instead.
"""

MIN_SEG = 4.0  # §3, §4: highlights shorter than 4.0 s are ignored in scoring
MIN_SEG_EMIT = 4.05  # emitted minimum: MIN_SEG plus float/rounding margin (ARCHITECTURE §5.6)
SCORING_BUDGET = 0.15  # §1, §3: highlights total <= 15% of duration (warning, soft penalty)
DURATION_TOL = 0.5  # §3: duration_sec within ±0.5 s of the manifest; segments may end at T + 0.5
MAX_ACTIONS = 2000  # §3: actions[] holds 0-2000 items
SHORT_VIDEO_T = MIN_SEG / SCORING_BUDGET  # 26.67 s: below this, 4 s and 15% conflict
TIME_DECIMALS = 2  # §3 note: timestamps in seconds, up to 2 decimal places
