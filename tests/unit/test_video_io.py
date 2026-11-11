"""Unit tests for frame <-> seconds conversion (PROGRESS.md P3: 25 / 30 / 59.94 fps)."""

import pytest

from rewind.video_io import (
    frame_to_sec,
    parse_ffprobe,
    parse_rate,
    sample_times,
    sec_to_frame,
)

NTSC_60 = 60000 / 1001  # "59.94" fps
NTSC_30 = 30000 / 1001  # "29.97" fps


# --- parse_rate -------------------------------------------------------------------


def test_parse_rate_ntsc_fraction_gives_59_94():
    # 60000 / 1001 = 59.9400599...
    assert parse_rate("60000/1001") == pytest.approx(59.94006, abs=1e-5)


def test_parse_rate_integer_string_gives_value():
    assert parse_rate("25") == pytest.approx(25.0, abs=1e-12)


@pytest.mark.parametrize("bad", ["0/1", "-30/1"])
def test_parse_rate_non_positive_raises(bad):
    with pytest.raises(ValueError):
        parse_rate(bad)


# --- frame_to_sec -----------------------------------------------------------------


def test_frame_to_sec_25fps_frame_50_is_2s():
    # 50 / 25 = 2.0 s
    assert frame_to_sec(50, 25.0) == pytest.approx(2.0, abs=1e-12)


def test_frame_to_sec_30fps_frame_45_is_1_5s():
    # 45 / 30 = 1.5 s
    assert frame_to_sec(45, 30.0) == pytest.approx(1.5, abs=1e-12)


def test_frame_to_sec_59_94fps_frame_60_is_1_001s():
    # 60 / (60000/1001) = 60 * 1001 / 60000 = 1.001 s
    assert frame_to_sec(60, NTSC_60) == pytest.approx(1.001, abs=1e-12)


def test_frame_to_sec_negative_frame_raises():
    with pytest.raises(ValueError):
        frame_to_sec(-1, 25.0)


# --- sec_to_frame -----------------------------------------------------------------


def test_sec_to_frame_25fps_inside_frame_50_gives_50():
    # frame 50 covers [2.00, 2.04) s at 25 fps; 2.039 * 25 = 50.975 -> 50
    assert sec_to_frame(2.039, 25.0) == 50


def test_sec_to_frame_25fps_next_boundary_gives_51():
    # 2.04 * 25 = 51.0 -> frame 51 starts exactly here
    assert sec_to_frame(2.04, 25.0) == 51


def test_sec_to_frame_59_94fps_exact_boundary_not_rounded_down():
    # 1.001 s is exactly the start of frame 60 at 59.94 fps (see above); float
    # arithmetic gives 59.99999..., which must still map to 60, not 59.
    assert sec_to_frame(1.001, NTSC_60) == 60


@pytest.mark.parametrize("fps", [25.0, 30.0, NTSC_30, NTSC_60])
def test_sec_to_frame_round_trip_returns_same_frame(fps):
    # frame -> its start time -> frame must be the identity, at every frame up to
    # ~1 hour of video (the longest EPIC video is ~33 min).
    for k in range(0, 216_000, 7):
        assert sec_to_frame(frame_to_sec(k, fps), fps) == k


def test_sec_to_frame_negative_time_raises():
    with pytest.raises(ValueError):
        sec_to_frame(-0.1, 25.0)


# --- sample_times -----------------------------------------------------------------


def test_sample_times_10s_at_2fps_gives_20_points_ending_9_5():
    # k / 2 for k = 0..19 -> 0.0, 0.5, ..., 9.5 (10.0 is not < 10.0)
    t = sample_times(10.0, 2.0)
    assert len(t) == 20
    assert t[0] == pytest.approx(0.0, abs=1e-12)
    assert t[-1] == pytest.approx(9.5, abs=1e-12)


def test_sample_times_10_2s_at_2fps_includes_10_0():
    # 10.0 < 10.2, so k = 20 is included: 21 points
    t = sample_times(10.2, 2.0)
    assert len(t) == 21
    assert t[-1] == pytest.approx(10.0, abs=1e-12)


def test_sample_times_all_inside_duration():
    t = sample_times(561.53, 4.0)
    # ceil(561.53 * 4) = ceil(2246.12) = 2247 points; last = 2246 / 4 = 561.5
    assert len(t) == 2247
    assert t[-1] == pytest.approx(561.5, abs=1e-12)
    assert t[-1] < 561.53


# --- parse_ffprobe ----------------------------------------------------------------


def _probe(fmt_duration="561.53", avg="60000/1001", r="60000/1001"):
    """A trimmed `ffprobe -show_format -show_streams` result: audio first, then video."""
    video = {
        "codec_type": "video",
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": avg,
        "r_frame_rate": r,
        "duration": "561.40",
    }
    fmt = {} if fmt_duration is None else {"duration": fmt_duration}
    return {"streams": [{"codec_type": "audio"}, video], "format": fmt}


def test_parse_ffprobe_reads_container_duration_fps_and_size():
    meta = parse_ffprobe(_probe())
    assert meta.duration_sec == pytest.approx(561.53, abs=1e-9)
    assert meta.fps == pytest.approx(60000 / 1001, abs=1e-9)
    assert (meta.width, meta.height) == (1920, 1080)


def test_parse_ffprobe_missing_container_duration_uses_stream_duration():
    meta = parse_ffprobe(_probe(fmt_duration=None))
    assert meta.duration_sec == pytest.approx(561.40, abs=1e-9)


def test_parse_ffprobe_avg_rate_zero_falls_back_to_r_frame_rate():
    meta = parse_ffprobe(_probe(avg="0/0", r="30/1"))
    assert meta.fps == pytest.approx(30.0, abs=1e-12)


def test_parse_ffprobe_prefers_avg_rate_over_r_frame_rate():
    # Variable-frame-rate phone video: r_frame_rate is the timebase-ish max (60),
    # avg_frame_rate is the true average (29.97). The average is the correct one.
    meta = parse_ffprobe(_probe(avg="30000/1001", r="60/1"))
    assert meta.fps == pytest.approx(30000 / 1001, abs=1e-9)


def test_parse_ffprobe_audio_only_raises():
    with pytest.raises(ValueError):
        parse_ffprobe({"streams": [{"codec_type": "audio"}], "format": {"duration": "10"}})


def test_parse_ffprobe_zero_duration_raises():
    with pytest.raises(ValueError):
        parse_ffprobe(_probe(fmt_duration="0.0"))
