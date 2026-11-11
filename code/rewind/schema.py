"""Output validator: kit rules R1–R13 (ARCHITECTURE.md §7). Pure: no file access.

``validate`` returns ``(errors, warnings)``. Each message starts with its rule id
(``"R6 actions[3].segment: ..."``) so callers and tests can match on it.

Two severity modes:
- ``strict=False``: the kit's severities, i.e. what the official validator would
  accept (it rejects errors and only warns on the rest).
- ``strict=True`` (what run.sh uses): also rejects anything that would lose points
  silently, i.e. highlights under 4 s (R12), more than 15% in total (R13, except for
  short videos, see ARCHITECTURE.md §5.6), labels or verbs/nouns outside the kit
  vocabulary (R4, R7) and timestamps with more than 2 decimals (R14).
"""

import math
from typing import Any

from rewind.constants import (
    DURATION_TOL,
    END_TOL,
    MAX_ACTIONS,
    MIN_SEG,
    MIN_SEG_EMIT,
    REQUIRED_KEYS,
    SCORING_BUDGET,
    SHORT_VIDEO_T,
    TIME_DECIMALS,
)
from rewind.vocab import Vocab

# Float slack for comparisons on values that went through JSON and rounding.
_EPS = 1e-9


def validate(
    doc: Any,
    vocab: Vocab,
    *,
    filename_stem: str | None = None,
    expected_duration: float | None = None,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Check one output document against the kit rules.

    filename_stem: the ``<video_id>`` of the file it will be written as (R2).
    expected_duration: the manifest duration, when known (R3; offline only).
    """
    errors: list[str] = []
    warnings: list[str] = []
    soft = errors if strict else warnings  # kit warnings that strict mode rejects

    if not isinstance(doc, dict):
        return [f"R1 <root>: expected a JSON object, got {type(doc).__name__}"], []
    missing = [k for k in REQUIRED_KEYS if k not in doc]
    if missing:
        errors.append(f"R1 <root>: missing key(s) {missing}")
    extra = sorted(set(doc) - set(REQUIRED_KEYS))
    if extra:
        errors.append(f"R1 <root>: unexpected key(s) {extra}")

    if "team_id" in doc and not _nonempty_str(doc["team_id"]):
        errors.append(f"R2 team_id: expected a non-empty string, got {doc['team_id']!r}")
    if "video_id" in doc:
        vid = doc["video_id"]
        if not _nonempty_str(vid):
            errors.append(f"R2 video_id: expected a non-empty string, got {vid!r}")
        elif filename_stem is not None and vid != filename_stem:
            errors.append(f"R2 video_id: {vid!r} does not match the file name {filename_stem!r}")

    duration = _check_duration(doc.get("duration_sec"), expected_duration, errors, "duration_sec")

    if "label" in doc:
        _check_label(doc["label"], vocab, errors, soft)

    if "actions" in doc:
        _check_actions(doc["actions"], duration, vocab, errors, soft)

    if "highlights" in doc:
        _check_highlights(doc["highlights"], duration, errors, warnings, soft, strict)

    return errors, warnings


def _nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and v != ""


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _too_many_decimals(x: float) -> bool:
    return abs(round(x, TIME_DECIMALS) - x) > _EPS


def _check_duration(d: Any, expected: float | None, errors: list[str], where: str) -> float | None:
    if d is None:
        return None
    if not _is_number(d) or d <= 0:
        errors.append(f"R3 {where}: expected a positive finite number, got {d!r}")
        return None
    if expected is not None and abs(d - expected) > DURATION_TOL + _EPS:
        errors.append(
            f"R3 {where}: {d} differs from the manifest duration {expected} by more than "
            f"{DURATION_TOL} s"
        )
    return float(d)


def _check_label(label: Any, vocab: Vocab, errors: list[str], soft: list[str]) -> None:
    if not isinstance(label, dict) or set(label) != {"name", "score"}:
        errors.append(f"R4 label: expected {{'name', 'score'}}, got {label!r}")
        return
    if not _nonempty_str(label["name"]):
        errors.append(f"R4 label.name: expected a non-empty string, got {label['name']!r}")
    elif label["name"] not in vocab.labels:
        soft.append(f"R4 label.name: {label['name']!r} is not one of the kit labels")
    if not _is_number(label["score"]) or not 0.0 <= label["score"] <= 1.0:
        errors.append(f"R4 label.score: expected a number in [0, 1], got {label['score']!r}")


def _check_segment(seg: Any, duration: float | None, where: str, rule: str) -> str | None:
    """Return an error message, or None if `seg` is a valid [start, end]."""
    if not isinstance(seg, list) or len(seg) != 2 or not all(_is_number(x) for x in seg):
        return f"{rule} {where}: expected [start, end] with two finite numbers, got {seg!r}"
    s, e = seg
    if s < 0:
        return f"{rule} {where}: start {s} < 0"
    if not s < e:
        return f"{rule} {where}: start {s} is not before end {e}"
    if duration is not None and e > duration + END_TOL + _EPS:
        return f"{rule} {where}: end {e} > duration_sec + {END_TOL} ({duration + END_TOL})"
    return None


def _check_actions(
    actions: Any, duration: float | None, vocab: Vocab, errors: list[str], soft: list[str]
) -> None:
    if not isinstance(actions, list):
        errors.append(f"R5 actions: expected a list, got {type(actions).__name__}")
        return
    if len(actions) > MAX_ACTIONS:
        errors.append(f"R5 actions: {len(actions)} items > {MAX_ACTIONS}")
    for i, a in enumerate(actions):
        where = f"actions[{i}]"
        if not isinstance(a, dict) or set(a) != {"segment", "verb", "noun", "score"}:
            errors.append(f"R6 {where}: expected keys {{segment, verb, noun, score}}, got {a!r}")
            continue
        msg = _check_segment(a["segment"], duration, f"{where}.segment", "R6")
        if msg:
            errors.append(msg)
        else:
            _check_decimals(a["segment"], f"{where}.segment", soft)
        for field, known in (("verb", vocab.verbs), ("noun", vocab.nouns)):
            v = a[field]
            if not _nonempty_str(v):
                errors.append(f"R7 {where}.{field}: expected a non-empty string, got {v!r}")
            elif v not in known:
                soft.append(f"R7 {where}.{field}: {v!r} is not in the kit vocabulary")
        if not _is_number(a["score"]) or not 0.0 <= a["score"] <= 1.0:
            errors.append(f"R8 {where}.score: expected a number in [0, 1], got {a['score']!r}")


def _check_decimals(seg: list, where: str, soft: list[str]) -> None:
    if any(_too_many_decimals(x) for x in seg):
        soft.append(f"R14 {where}: {seg} has more than {TIME_DECIMALS} decimal places")


def _check_highlights(
    highlights: Any,
    duration: float | None,
    errors: list[str],
    warnings: list[str],
    soft: list[str],
    strict: bool,
) -> None:
    if not isinstance(highlights, list) or not highlights:
        errors.append(f"R9 highlights: expected a non-empty list, got {highlights!r}")
        return
    segs: list[tuple[float, float]] = []
    for i, h in enumerate(highlights):
        where = f"highlights[{i}]"
        if not isinstance(h, dict) or set(h) != {"segment"}:
            errors.append(f"R9 {where}: expected {{'segment': [start, end]}}, got {h!r}")
            continue
        msg = _check_segment(h["segment"], duration, f"{where}.segment", "R9")
        if msg:
            errors.append(msg)
            continue
        _check_decimals(h["segment"], f"{where}.segment", soft)
        segs.append((h["segment"][0], h["segment"][1]))
    if len(segs) != len(highlights):
        return  # ordering and length rules need every segment to be valid

    for i in range(1, len(segs)):
        (s0, e0), (s1, _) = segs[i - 1], segs[i]
        if s1 < s0:
            errors.append(f"R10 highlights[{i}]: starts at {s1}, before highlights[{i - 1}]")
        elif s1 <= e0:
            # "strictly non-overlapping": touching (s1 == e0) counts as overlap here.
            errors.append(f"R10 highlights[{i}]: starts at {s1}, not after previous end {e0}")

    lengths = [e - s for s, e in segs]
    if not any(L >= MIN_SEG - _EPS for L in lengths):
        errors.append(f"R11 highlights: none is >= {MIN_SEG} s (lengths {lengths})")
    for i, L in enumerate(lengths):
        if L < MIN_SEG - _EPS:
            soft.append(f"R12 highlights[{i}]: {L:.3f} s < {MIN_SEG} s is ignored in scoring")

    if duration is not None:
        total = sum(lengths)
        if total > SCORING_BUDGET * duration + _EPS:
            msg = (
                f"R13 highlights: total {total:.3f} s > {SCORING_BUDGET:.0%} of "
                f"{duration} s ({SCORING_BUDGET * duration:.3f} s)"
            )
            # A video under SHORT_VIDEO_T can't fit one 4 s window in the budget: the
            # kit makes R11 pass/fail and R13 a warning, so this stays a warning.
            short_ok = (
                duration < SHORT_VIDEO_T and len(segs) == 1 and lengths[0] <= MIN_SEG_EMIT + _EPS
            )
            (soft if strict and not short_ok else warnings).append(msg)
