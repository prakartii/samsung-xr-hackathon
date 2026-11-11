"""VideoResult → the kit's JSON document (ARCHITECTURE.md §7). Pure.

Byte-stable: same result → same string. Keys in kit order; times rounded to 2
decimals; scores to 4; actions sorted; actions that collapse under rounding dropped.
"""

import json
import math
from typing import Any

from rewind.constants import MAX_ACTIONS, TIME_DECIMALS
from rewind.types import VideoResult

SCORE_DECIMALS = 4


def _score(x: float) -> float:
    return round(min(1.0, max(0.0, float(x))), SCORE_DECIMALS)


def _t(x: float) -> float:
    return round(float(x), TIME_DECIMALS)


def to_document(result: VideoResult) -> dict[str, Any]:
    actions = []
    for (s, e), verb, noun, score in result.actions:
        if not (math.isfinite(s) and math.isfinite(e) and math.isfinite(score)):
            continue
        # Clamp to [0, T] before rounding, so one stray action can't break R6.
        s2, e2 = _t(max(0.0, s)), _t(min(e, result.duration_sec))
        if s2 < e2:
            actions.append(
                {"segment": [s2, e2], "verb": verb, "noun": noun, "score": _score(score)}
            )
    # Highest score first (what mAP ranks on), then time, then names: a total order.
    actions.sort(key=lambda a: (-a["score"], a["segment"], a["verb"], a["noun"]))
    name, label_score = result.label
    return {
        "team_id": result.team_id,
        "video_id": result.video_id,
        "duration_sec": _t(result.duration_sec),
        "label": {"name": name, "score": _score(label_score)},
        "actions": actions[:MAX_ACTIONS],
        "highlights": [{"segment": [_t(s), _t(e)]} for s, e in sorted(result.highlights)],
    }


def dumps(doc: dict[str, Any]) -> str:
    """Serialize a document deterministically (one space indent, trailing newline)."""
    return json.dumps(doc, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
