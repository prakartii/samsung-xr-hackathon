"""postprocess: hand-made cases with exact expected windows (ARCHITECTURE.md §5.6).

Default setup: T = 100 s, budget_frac 0.145 → B = 14.5 s, merge_gap 0.5 s, min_seg mode.
"""

import numpy as np
import pytest

from rewind.postprocess import postprocess

KW = {"budget_frac": 0.145, "merge_gap": 0.5, "short_video_mode": "min_seg"}


def run(segments, values=None, duration=100.0, **overrides):
    values = [1.0] * len(segments) if values is None else values
    return postprocess(segments, values, duration, **{**KW, **overrides})


def test_postprocess_segment_before_zero_is_clamped():
    # (-2, 5) → (0, 5): 5 s ≥ 4.05 kept.
    assert run([(-2.0, 5.0)]) == ((0.0, 5.0),)


def test_postprocess_overlapping_segments_are_merged():
    assert run([(10.0, 15.0), (14.0, 18.0)]) == ((10.0, 18.0),)


def test_postprocess_gap_below_merge_gap_is_merged():
    # gap 15.3 - 15.0 = 0.3 < 0.5
    assert run([(10.0, 15.0), (15.3, 19.0)]) == ((10.0, 19.0),)


def test_postprocess_gap_equal_to_merge_gap_is_kept_separate():
    # gap 15.5 - 15.0 = 0.5, not < 0.5
    assert run([(10.0, 15.0), (15.5, 20.0)]) == ((10.0, 15.0), (15.5, 20.0))


def test_postprocess_touching_segments_merge_even_with_zero_merge_gap():
    # R10 forbids touching, so (10, 15) and (15, 20) must become one segment.
    assert run([(10.0, 15.0), (15.0, 20.0)], merge_gap=0.0) == ((10.0, 20.0),)


def test_postprocess_fragment_under_min_emit_is_dropped():
    # 14.0 - 10.0 = 4.0 < 4.05 (could round below 4 s), so it goes; (30, 35) stays.
    assert run([(10.0, 14.0), (30.0, 35.0)]) == ((30.0, 35.0),)


def test_postprocess_over_budget_trims_lowest_value_segment():
    # Total 10 + 10 = 20 > 14.5, excess 5.5. Segment (10, 20) has the lower value (1 < 2)
    # and 10 - 4.05 = 5.95 s of room ≥ 5.5, so it becomes (15.5, 20). Total 14.5.
    assert run([(10.0, 20.0), (40.0, 50.0)], [1.0, 2.0]) == ((15.5, 20.0), (40.0, 50.0))


def test_postprocess_over_budget_drops_segment_already_at_minimum():
    # Total 4.05 + 12 = 16.05 > 14.5. The lower-value segment has no room above 4.05,
    # so it is dropped; 12 ≤ 14.5 remains.
    assert run([(10.0, 14.05), (40.0, 52.0)], [0.0, 1.0]) == ((40.0, 52.0),)


def test_postprocess_trim_never_goes_below_min_emit():
    # Total 10 + 10.5 = 20.5 > 14.5, excess 6.0. The lower-value (10, 20) has only
    # 10 - 4.05 = 5.95 s of room: it shrinks to (15.95, 20) and the total is still
    # 4.05 + 10.5 = 14.55 > 14.5, so it is dropped. Shrinking it to 4.0 s instead
    # would fit exactly, but a 4.0 s window can round to 3.99 s.
    assert run([(10.0, 20.0), (40.0, 50.5)], [1.0, 2.0]) == ((40.0, 50.5),)


@pytest.mark.parametrize(
    ("profile_values", "expected"),
    [
        # T = 50 → B = 7.25; (10, 20) is 10 s, excess 2.75.
        ([0.0, 1.0], (12.75, 20.0)),  # value rises with time: trim the start
        ([1.0, 0.0], (10.0, 17.25)),  # value falls with time: trim the end
    ],
)
def test_postprocess_profile_picks_lower_value_end_to_trim(profile_values, expected):
    profile = (np.array([0.0, 50.0]), np.array(profile_values))

    assert run([(10.0, 20.0)], duration=50.0, profile=profile) == (expected,)


def test_postprocess_empty_input_gives_one_window_in_the_middle():
    # Centre 50: (47.975, 52.025) rounded inward → (47.98, 52.02).
    assert run([]) == ((47.98, 52.02),)


def test_postprocess_empty_input_with_profile_centres_on_its_peak():
    profile = (np.array([0.0, 80.0, 100.0]), np.array([0.0, 1.0, 0.0]))

    # Peak at 80: (77.975, 82.025) → (77.98, 82.02).
    assert run([], profile=profile) == ((77.98, 82.02),)


def test_postprocess_budget_below_one_window_keeps_one_at_best_candidate():
    # T = 27.5 ≥ 27.0 (not short) but B = 0.145 · 27.5 = 3.99 < 4.05: nothing fits.
    # Keep one 4.05 s window centred on the input's centre 10: (7.975, 12.025) →
    # (7.98, 12.02). 4.04 ≤ 0.15 · 27.5 = 4.125, so the kit budget still holds.
    assert run([(5.0, 15.0)], duration=27.5) == ((7.98, 12.02),)


def test_postprocess_rounds_start_up_and_end_down():
    assert run([(10.004, 20.006)]) == ((10.01, 20.0),)


def test_postprocess_short_video_min_seg_mode_gives_one_min_window():
    # T = 12 < 27: one 4.05 s window centred on the best segment (6, 9) → 7.5:
    # (5.475, 9.525) → (5.48, 9.52).
    assert run([(6.0, 9.0)], duration=12.0) == ((5.48, 9.52),)


def test_postprocess_short_video_within_budget_mode_gives_fifteen_percent():
    # 0.15 · 12 = 1.8 s centred on 7.5: (6.6, 8.4).
    result = run([(6.0, 9.0)], duration=12.0, short_video_mode="within_budget")

    assert result == (pytest.approx((6.6, 8.4), abs=1e-9),)


def test_postprocess_short_video_without_input_centres_on_middle():
    # Centre 6: (3.975, 8.025) → (3.98, 8.02).
    assert run([], duration=12.0) == ((3.98, 8.02),)


def test_postprocess_non_finite_segment_is_ignored():
    assert run([(float("nan"), 5.0), (30.0, 35.0)]) == ((30.0, 35.0),)


def test_postprocess_values_length_mismatch_raises():
    with pytest.raises(ValueError, match="2 segments but 1 values"):
        run([(1.0, 6.0), (10.0, 15.0)], [1.0])


def test_postprocess_short_video_centres_on_the_highest_value_segment():
    # T = 12: (1, 2) has value .1, (8, 9) has .9 → centre 8.5: (6.475, 10.525) →
    # (6.48, 10.52).
    assert run([(1.0, 2.0), (8.0, 9.0)], [0.1, 0.9], duration=12.0) == ((6.48, 10.52),)


def test_postprocess_trim_tie_goes_to_the_later_segment():
    # Equal values: the later segment (40, 50) is trimmed by the excess 20 - 14.5 = 5.5
    # from its start (no profile) → (45.5, 50).
    assert run([(10.0, 20.0), (40.0, 50.0)], [1.0, 1.0]) == ((10.0, 20.0), (45.5, 50.0))


def test_postprocess_merged_segment_keeps_the_max_value():
    # (10, 15) v .1 and (14, 18) v .9 merge into (10, 18) with value .9; (40, 50) v .5
    # is now the lowest and absorbs the excess 18 - 14.5 = 3.5 → (43.5, 50).
    result = run([(10.0, 15.0), (14.0, 18.0), (40.0, 50.0)], [0.1, 0.9, 0.5])

    assert result == ((10.0, 18.0), (43.5, 50.0))


def test_postprocess_budget_uses_the_written_duration():
    # T = 36.534 is written as 36.53; at budget_frac 0.15 the cap is 0.15 · 36.53 =
    # 5.4795 s, not 0.15 · 36.534 = 5.4801 s. (0, 20) is trimmed to (14.5205, 20) →
    # (14.53, 20.0), 5.47 s, which passes the validator's check on 36.53.
    result = run([(0.0, 20.0)], duration=36.534, budget_frac=0.15)

    assert result == ((14.53, 20.0),)
    assert 20.0 - 14.53 <= 0.15 * 36.53


def test_postprocess_video_between_3_5_and_4_s_gets_a_4_s_highlight():
    # Kit: ends may reach T + 0.5, so for T = 3.8 one (0, 4.0) highlight is valid.
    assert run([], duration=3.8) == ((0.0, 4.0),)
