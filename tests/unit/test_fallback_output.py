"""fallback (uniform safe output) and output (VideoResult → kit JSON)."""

import json

import pytest

from rewind.fallback import fallback_result, uniform_windows
from rewind.output import dumps, to_document
from rewind.types import VideoResult

KW = {"budget_frac": 0.145, "merge_gap": 0.5, "short_video_mode": "min_seg"}


def test_uniform_windows_count_and_centres():
    # B = 0.145 · 100 = 14.5 s → floor(14.5 / 4.05) = 3 windows, centres at
    # (i + 0.5) · 100 / 3 = 16.67, 50.0, 83.33, each ± 2.025.
    windows = uniform_windows(100.0, 0.145)

    assert [(s + e) / 2 for s, e in windows] == pytest.approx([50 / 3, 50.0, 250 / 3], abs=1e-9)
    assert all(e - s == pytest.approx(4.05, abs=1e-9) for s, e in windows)


def test_fallback_result_has_rounded_uniform_highlights_and_no_actions():
    r = fallback_result("T07", "P01_11", 100.0, **KW)

    # (14.6417, 18.6917) → (14.65, 18.69); (47.975, 52.025) → (47.98, 52.02);
    # (81.3083, 85.3583) → (81.31, 85.35).
    assert r.highlights == ((14.65, 18.69), (47.98, 52.02), (81.31, 85.35))
    assert r.actions == ()
    assert r.label == ("other", 0.0)


def make_result(**changes):
    base = {
        "team_id": "T07",
        "video_id": "P01_11",
        "duration_sec": 100.004,
        "label": ("cooking", 1.0),
        "actions": (((12.4, 18.9), "open", "fridge", 0.8),),
        "highlights": ((10.0, 14.05),),
    }
    return VideoResult(**{**base, **changes})


def test_to_document_keys_follow_kit_order():
    assert list(to_document(make_result())) == [
        "team_id",
        "video_id",
        "duration_sec",
        "label",
        "actions",
        "highlights",
    ]


def test_to_document_rounds_duration_to_two_decimals():
    assert to_document(make_result())["duration_sec"] == pytest.approx(100.0, abs=1e-12)


def test_to_document_sorts_actions_by_score_then_time():
    actions = (
        ((30.0, 35.0), "cut", "onion", 0.5),
        ((10.0, 15.0), "wash", "knife", 0.9),
        ((5.0, 8.0), "open", "fridge", 0.5),
    )

    doc = to_document(make_result(actions=actions))

    assert [a["verb"] for a in doc["actions"]] == ["wash", "open", "cut"]


def test_to_document_drops_action_that_collapses_when_rounded():
    # (1.001, 1.004) rounds to (1.0, 1.0): no longer start < end.
    actions = (((1.001, 1.004), "cut", "onion", 0.5),)

    assert to_document(make_result(actions=actions))["actions"] == []


@pytest.mark.parametrize(("raw", "clamped"), [(1.7, 1.0), (-0.2, 0.0), (0.123456, 0.1235)])
def test_to_document_clamps_and_rounds_scores(raw, clamped):
    doc = to_document(make_result(actions=(((1.0, 2.0), "cut", "onion", raw),)))

    assert doc["actions"][0]["score"] == pytest.approx(clamped, abs=1e-12)


def test_to_document_caps_actions_at_2000_keeping_highest_scores():
    actions = tuple(((float(i), i + 1.0), "cut", "onion", i / 3000) for i in range(2001))

    doc = to_document(make_result(actions=actions))

    assert len(doc["actions"]) == 2000
    assert min(a["score"] for a in doc["actions"]) == pytest.approx(round(1 / 3000, 4), abs=1e-12)


def test_dumps_is_deterministic_json_with_trailing_newline():
    doc = to_document(make_result())

    text = dumps(doc)

    assert text == dumps(to_document(make_result()))
    assert text.endswith("}\n")
    assert json.loads(text) == doc
