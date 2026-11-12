"""features/audio.py: loudness (RMS) and spectral flux on the dense time grid (P6)."""

import numpy as np
import pytest

from rewind.features.audio import rms_at, spectral_flux_at

SR = 8000


def test_rms_at_sine_is_amplitude_over_root_two():
    # 0.5 * sin(2 pi 100 t): RMS over whole periods = 0.5 / sqrt(2) = 0.35355
    t = np.arange(SR * 2) / SR
    x = 0.5 * np.sin(2 * np.pi * 100 * t)
    out = rms_at(x, SR, np.array([0.5, 1.0, 1.5]), win=0.25)
    assert out == pytest.approx([0.5 / np.sqrt(2)] * 3, abs=1e-3)


def test_rms_at_silence_then_tone_is_zero_then_loud():
    # first second silent, second second a 0.4 amplitude sine (RMS 0.2828)
    t = np.arange(SR * 2) / SR
    x = np.where(t < 1.0, 0.0, 0.4 * np.sin(2 * np.pi * 200 * t))
    out = rms_at(x, SR, np.array([0.5, 1.5]), win=0.25)
    assert out == pytest.approx([0.0, 0.4 / np.sqrt(2)], abs=1e-3)


def test_rms_at_window_past_the_end_uses_the_samples_left():
    # 1 s of constant 0.3: a window centred at t = 1.0 holds only the last half-window
    x = np.full(SR, 0.3)
    assert rms_at(x, SR, np.array([1.0]), win=0.5) == pytest.approx([0.3], abs=1e-9)


def test_rms_at_no_samples_in_window_is_zero():
    assert rms_at(np.ones(SR), SR, np.array([5.0]), win=0.25) == pytest.approx([0.0], abs=1e-12)


def test_spectral_flux_peaks_at_an_onset():
    # silence, then a tone from 1.0 s: flux at the onset (1.0) > in steady tone (1.6)
    # and > in silence (0.4), where it is 0.
    t = np.arange(SR * 2) / SR
    x = np.where(t < 1.0, 0.0, 0.5 * np.sin(2 * np.pi * 440 * t))
    out = spectral_flux_at(x, SR, np.array([0.4, 1.0, 1.6]), win=0.25)
    assert out[0] == pytest.approx(0.0, abs=1e-9)
    assert out[1] > 10 * max(out[2], 1e-9)


def test_audio_loader_caches_a_video_without_audio(tmp_path):
    # A PyAV-made video with no audio stream: None, and the cache remembers it, so
    # the second call needs no MP4 at all.
    from fixtures.videos import write_index_video

    from rewind.analyze import audio_loader
    from rewind.config import load_config
    from tests.conftest import DEFAULT_CONFIG

    cfg = load_config(DEFAULT_CONFIG)
    video = write_index_video(tmp_path / "v.avi", n_frames=8, fps=4.0)
    t = np.arange(8) / 4.0
    assert audio_loader(video, "fp1", cfg, tmp_path / "cache")(t) is None
    video.unlink()
    assert audio_loader(video, "fp1", cfg, tmp_path / "cache")(t) is None


def test_audio_loader_without_video_or_cache_entry_is_none(tmp_path):
    from rewind.analyze import audio_loader
    from rewind.config import load_config
    from tests.conftest import DEFAULT_CONFIG

    cfg = load_config(DEFAULT_CONFIG)
    load = audio_loader(tmp_path / "missing.mp4", "fp2", cfg, tmp_path / "cache")
    assert load(np.arange(4) / 4.0) is None
