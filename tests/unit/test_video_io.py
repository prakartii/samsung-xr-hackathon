"""parse_probe: ffprobe JSON → ProbeInfo (pure)."""

import pytest

from rewind.video_io import VideoError, parse_probe


def probe(fmt_dur="30.5", st_dur="30.4", rate="30000/1001"):
    return {
        "streams": [{"width": 1920, "height": 1080, "avg_frame_rate": rate, "duration": st_dur}],
        "format": {"duration": fmt_dur},
    }


def test_parse_probe_prefers_container_duration():
    assert parse_probe(probe()).duration_sec == pytest.approx(30.5, abs=1e-9)


def test_parse_probe_falls_back_to_stream_duration():
    assert parse_probe(probe(fmt_dur=None)).duration_sec == pytest.approx(30.4, abs=1e-9)


def test_parse_probe_reads_fractional_frame_rate():
    # 30000 / 1001 = 29.97...
    assert parse_probe(probe()).fps == pytest.approx(29.97003, abs=1e-5)


def test_parse_probe_zero_frame_rate_denominator_gives_zero_fps():
    assert parse_probe(probe(rate="0/0")).fps == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "info",
    [
        {"streams": [], "format": {"duration": "10"}},
        probe(fmt_dur=None, st_dur=None),
        probe(fmt_dur="N/A", st_dur=None),
        probe(fmt_dur="0"),
    ],
)
def test_parse_probe_unusable_input_raises_video_error(info):
    with pytest.raises(VideoError):
        parse_probe(info)
