"""Audio activity signals on the dense grid: loudness (RMS) and spectral flux (P6).

Ablation candidates for the FPVSum importance proxy (PROGRESS P6: "keep only if CV
improves"). Pure functions on a mono sample array, plus `decode_mono` at the edge
(PyAV). Each value at time t summarises the samples in [t - win/2, t + win/2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FLUX_NFFT = 512  # STFT frame for spectral flux (64 ms at 8 kHz)
RMS_FLOOR = 1e-4  # log(rms + floor): silence maps to a finite loudness
AUDIO_VERSION = "audio-v1"  # bump when the extraction below changes


def _window(t: float, sr: int, win: float, n: int) -> tuple[int, int]:
    lo = max(0, round((t - win / 2) * sr))
    hi = min(n, round((t + win / 2) * sr))
    return lo, max(lo, hi)


def rms_at(x: np.ndarray, sr: int, t: np.ndarray, win: float) -> np.ndarray:
    """(N,) root mean square of the samples around each t (0 where there are none)."""
    x = np.asarray(x, dtype=np.float64)
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    out = np.zeros(len(t))
    for i, ti in enumerate(t):
        lo, hi = _window(float(ti), sr, win, len(x))
        if hi > lo:
            out[i] = np.sqrt((c[hi] - c[lo]) / (hi - lo))
    return out


def spectral_flux_at(x: np.ndarray, sr: int, t: np.ndarray, win: float) -> np.ndarray:
    """(N,) mean positive spectral flux, sum_k max(0, |X_j,k| - |X_j-1,k|), over the
    STFT frames (FLUX_NFFT, hop FLUX_NFFT / 2, Hann) starting inside each window."""
    x = np.asarray(x, dtype=np.float64)
    hop = FLUX_NFFT // 2
    n_frames = max(0, (len(x) - FLUX_NFFT) // hop + 1)
    if n_frames < 2:
        return np.zeros(len(t))
    idx = np.arange(FLUX_NFFT)[None, :] + hop * np.arange(n_frames)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * np.hanning(FLUX_NFFT), axis=1))
    flux = np.concatenate([[0.0], np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)])
    starts = hop * np.arange(n_frames)
    out = np.zeros(len(t))
    for i, ti in enumerate(t):
        lo, hi = _window(float(ti), sr, win, len(x))
        sel = (starts >= lo) & (starts < hi)
        if sel.any():
            out[i] = flux[sel].mean()
    return out


def decode_mono(path: Path, sr: int) -> np.ndarray | None:
    """The first audio stream as mono float samples at ``sr`` Hz; None if no audio."""
    import av

    with av.open(str(path)) as c:
        if not c.streams.audio:
            return None
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        chunks = []
        for frame in c.decode(c.streams.audio[0]):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def audio_cache_version(sample_rate: int, window: float) -> str:
    return f"{AUDIO_VERSION}-sr{sample_rate}-win{window}-nfft{FLUX_NFFT}"


def audio_signals(path: Path, t: np.ndarray, sample_rate: int, window: float):
    """{"rms", "flux"} on the time grid ``t``; None if the video has no audio stream."""
    x = decode_mono(path, sample_rate)
    if x is None:
        return None
    return {
        "rms": rms_at(x, sample_rate, t, window),
        "flux": spectral_flux_at(x, sample_rate, t, window),
    }
