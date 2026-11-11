"""rewind.schema.validate: one passing document, one failing case per kit rule.

Base document: T = 100 s, two highlights of 4.05 s and 5.0 s (total 9.05 s, under the
15 s budget), one valid action, label "cooking".
"""

import copy
import json

import pytest

from rewind.schema import validate
from rewind.vocab import load_vocab

VOCAB = load_vocab()
BASE = {
    "team_id": "T07",
    "video_id": "P01_11",
    "duration_sec": 100.0,
    "label": {"name": "cooking", "score": 0.9},
    "actions": [{"segment": [12.4, 18.9], "verb": "open", "noun": "fridge", "score": 0.8}],
    "highlights": [{"segment": [10.0, 14.05]}, {"segment": [40.0, 45.0]}],
}


def check(mutate=None, *, strict=False, stem="P01_11", expected=None):
    doc = copy.deepcopy(BASE)
    if mutate:
        mutate(doc)
    return validate(doc, VOCAB, filename_stem=stem, expected_duration=expected, strict=strict)


def rules(messages):
    return sorted({m.split(" ", 1)[0] for m in messages})


# --- passing ---------------------------------------------------------------


@pytest.mark.parametrize("strict", [False, True])
def test_validate_valid_document_has_no_errors_or_warnings(strict):
    assert check(strict=strict) == ([], [])


def test_validate_kit_example_files_pass_kit_rules():
    # kit/P02_109.json is a reference output; it must pass at kit severity.
    from tests.conftest import KIT_DIR

    doc = json.loads((KIT_DIR / "P02_109.json").read_text(encoding="utf-8"))
    errors, _ = validate(doc, VOCAB, filename_stem="P02_109", expected_duration=512.3)

    assert errors == []


def test_validate_empty_actions_are_allowed():
    # Kit §3: actions "May be [] for FPVSum videos".
    assert check(lambda d: d.update(actions=[])) == ([], [])


# --- R1 keys -------------------------------------------------------------------


def test_validate_missing_key_is_r1_error():
    errors, _ = check(lambda d: d.pop("label"))
    assert rules(errors) == ["R1"]


def test_validate_extra_top_level_key_is_r1_error():
    errors, _ = check(lambda d: d.update(model="x"))
    assert rules(errors) == ["R1"]


def test_validate_non_object_document_is_r1_error():
    errors, _ = validate([], VOCAB)
    assert rules(errors) == ["R1"]


# --- R2 ids --------------------------------------------------------------------


def test_validate_empty_team_id_is_r2_error():
    errors, _ = check(lambda d: d.update(team_id=""))
    assert rules(errors) == ["R2"]


def test_validate_video_id_not_matching_filename_is_r2_error():
    errors, _ = check(stem="P01_12")
    assert rules(errors) == ["R2"]


def test_validate_non_string_video_id_is_r2_error():
    errors, _ = check(lambda d: d.update(video_id=11), stem=None)
    assert rules(errors) == ["R2"]


# --- R3 duration ------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -5.0, "100", True, float("nan")])
def test_validate_bad_duration_is_r3_error(bad):
    errors, _ = check(lambda d: d.update(duration_sec=bad))
    assert "R3" in rules(errors)


def test_validate_duration_more_than_half_second_off_manifest_is_r3_error():
    # |100.0 - 100.51| = 0.51 > 0.5
    errors, _ = check(expected=100.51)
    assert rules(errors) == ["R3"]


def test_validate_duration_exactly_half_second_off_manifest_passes():
    # |100.0 - 100.5| = 0.5, inside the ±0.5 s tolerance.
    assert check(expected=100.5) == ([], [])


# --- R4 label ----------------------------------------------------------------


@pytest.mark.parametrize(
    "label", [{"name": "cooking", "score": 1.1}, "cooking", {"name": "", "score": 0.5}]
)
def test_validate_malformed_label_is_r4_error(label):
    errors, _ = check(lambda d: d.update(label=label))
    assert rules(errors) == ["R4"]


def test_validate_unknown_label_name_is_r4_warning_at_kit_severity():
    errors, warnings = check(lambda d: d["label"].update(name="skiing"))
    assert (errors, rules(warnings)) == ([], ["R4"])


def test_validate_unknown_label_name_is_r4_error_when_strict():
    errors, _ = check(lambda d: d["label"].update(name="skiing"), strict=True)
    assert rules(errors) == ["R4"]


# --- R5–R8 actions -----------------------------------------------------------


def test_validate_more_than_2000_actions_is_r5_error():
    errors, _ = check(lambda d: d.update(actions=d["actions"] * 2001))
    assert rules(errors) == ["R5"]


def test_validate_exactly_2000_actions_passes():
    assert check(lambda d: d.update(actions=d["actions"] * 2000)) == ([], [])


@pytest.mark.parametrize(
    "segment",
    [
        [-0.1, 5.0],  # start < 0
        [5.0, 5.0],  # start == end
        [6.0, 5.0],  # start > end
        [90.0, 100.51],  # end > T + 0.5 = 100.5
        [1.0, 2.0, 3.0],  # three numbers
        [1.0, "2"],  # not a number
    ],
)
def test_validate_bad_action_segment_is_r6_error(segment):
    errors, _ = check(lambda d: d["actions"][0].update(segment=segment))
    assert rules(errors) == ["R6"]


def test_validate_action_ending_at_duration_plus_tolerance_passes():
    # Kit §3: end <= duration_sec + 0.5 = 100.5.
    assert check(lambda d: d["actions"][0].update(segment=[95.0, 100.5])) == ([], [])


def test_validate_action_with_missing_field_is_r6_error():
    errors, _ = check(lambda d: d["actions"][0].pop("score"))
    assert rules(errors) == ["R6"]


def test_validate_unknown_verb_is_r7_warning_at_kit_severity():
    errors, warnings = check(lambda d: d["actions"][0].update(verb="teleport"))
    assert (errors, rules(warnings)) == ([], ["R7"])


def test_validate_unknown_noun_is_r7_error_when_strict():
    errors, _ = check(lambda d: d["actions"][0].update(noun="spaceship"), strict=True)
    assert rules(errors) == ["R7"]


@pytest.mark.parametrize("score", [-0.1, 1.01, "0.5", None])
def test_validate_bad_action_score_is_r8_error(score):
    errors, _ = check(lambda d: d["actions"][0].update(score=score))
    assert rules(errors) == ["R8"]


# --- R9–R13 highlights ------------------------------------------------------


@pytest.mark.parametrize("highlights", [[], None, [{"segment": [1.0, 6.0], "importance": 1}]])
def test_validate_malformed_highlights_is_r9_error(highlights):
    errors, _ = check(lambda d: d.update(highlights=highlights))
    assert rules(errors) == ["R9"]


def test_validate_highlight_past_duration_is_r9_error():
    # end 101.0 > T + 0.5 = 100.5
    errors, _ = check(lambda d: d["highlights"].append({"segment": [96.0, 101.0]}))
    assert rules(errors) == ["R9"]


def test_validate_unsorted_highlights_is_r10_error_naming_the_order():
    # [40, 45] then [10, 14.05]: the second starts before the first.
    errors, _ = check(lambda d: d["highlights"].reverse())
    assert errors == ["R10 highlights[1]: starts at 10.0, before highlights[0]"]


def test_validate_overlapping_highlights_is_r10_error():
    # [10, 14.05] and [12, 17]: 12 < 14.05
    errors, _ = check(
        lambda d: d.update(highlights=[{"segment": [10.0, 14.05]}, {"segment": [12.0, 17.0]}])
    )
    assert rules(errors) == ["R10"]


def test_validate_touching_highlights_is_r10_error():
    # "strictly non-overlapping": [10, 14.05] then [14.05, 19.05] share 14.05.
    errors, _ = check(
        lambda d: d.update(highlights=[{"segment": [10.0, 14.05]}, {"segment": [14.05, 19.05]}])
    )
    assert rules(errors) == ["R10"]


def test_validate_no_highlight_of_4s_is_r11_error_even_at_kit_severity():
    # Lengths 3.99 and 2.0: none >= 4.0.
    errors, _ = check(
        lambda d: d.update(highlights=[{"segment": [10.0, 13.99]}, {"segment": [40.0, 42.0]}])
    )
    assert "R11" in rules(errors)


def test_validate_short_highlight_beside_a_valid_one_is_r12_warning():
    # [10, 14.05] is 4.05 s; [40, 43] is 3.0 s < 4.0 and is ignored in scoring.
    errors, warnings = check(lambda d: d["highlights"][1].update(segment=[40.0, 43.0]))
    assert (errors, rules(warnings)) == ([], ["R12"])


def test_validate_short_highlight_is_r12_error_when_strict():
    errors, _ = check(lambda d: d["highlights"][1].update(segment=[40.0, 43.0]), strict=True)
    assert rules(errors) == ["R12"]


def over_budget(d):
    # 4.05 + 12.0 = 16.05 s > 0.15 * 100 = 15.0 s
    d["highlights"][1]["segment"] = [40.0, 52.0]


def test_validate_over_budget_is_r13_warning_at_kit_severity():
    errors, warnings = check(over_budget)
    assert (errors, rules(warnings)) == ([], ["R13"])


def test_validate_over_budget_is_r13_error_when_strict():
    errors, _ = check(over_budget, strict=True)
    assert rules(errors) == ["R13"]


def short_video(length):
    def mutate(d):
        # T = 12 s: 15% is 1.8 s, so no 4 s highlight fits the budget.
        d.update(duration_sec=12.0, actions=[], highlights=[{"segment": [3.0, 3.0 + length]}])

    return mutate


def test_validate_short_video_single_min_highlight_passes_strict_with_warning():
    errors, warnings = check(short_video(4.05), strict=True)
    assert (errors, rules(warnings)) == ([], ["R13"])


def test_validate_short_video_longer_highlight_is_r13_error_when_strict():
    # 6.0 s is more than the minimum needed, so the exemption doesn't apply.
    errors, _ = check(short_video(6.0), strict=True)
    assert rules(errors) == ["R13"]


# --- R14 decimals ------------------------------------------------------------


def test_validate_three_decimal_timestamp_is_r14_warning_at_kit_severity():
    errors, warnings = check(lambda d: d["highlights"][0].update(segment=[10.0, 14.055]))
    assert (errors, rules(warnings)) == ([], ["R14"])


def test_validate_three_decimal_timestamp_is_r14_error_when_strict():
    errors, _ = check(lambda d: d["actions"][0].update(segment=[12.4, 18.905]), strict=True)
    assert rules(errors) == ["R14"]
