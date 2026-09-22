from __future__ import annotations

import numpy as np

from config import AUDIO


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multichannel audio to mono by averaging channels."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Audio is empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Audio contains NaN or Inf values.")
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        return arr.mean(axis=1, dtype=np.float32).astype(np.float32)
    raise ValueError("Unsupported audio shape; expected 1D or 2D array.")


def resample_audio(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to a target sample rate without changing the content."""
    if original_sr == target_sr or audio.size == 0:
        return np.asarray(audio, dtype=np.float32)
    if original_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive integers.")

    if audio.size < 2:
        return np.asarray(audio, dtype=np.float32)

    ratio = target_sr / original_sr
    target_len = max(1, int(np.ceil(len(audio) * ratio)))
    x_old = np.linspace(0, len(audio) - 1, len(audio), dtype=np.float32)
    x_new = np.linspace(0, len(audio) - 1, target_len, dtype=np.float32)
    resampled = np.interp(x_new, x_old, audio).astype(np.float32)
    return resampled


def compute_rms(audio: np.ndarray) -> float:
    """Return the root-mean-square of a one-dimensional waveform."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def dominant_frequency(audio: np.ndarray, sample_rate: int = AUDIO.SAMPLE_RATE) -> float:
    """Return the strongest non-DC frequency component in an audio window."""
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size < 4 or not np.all(np.isfinite(arr)):
        return 0.0
    spectrum = np.abs(np.fft.rfft(arr * np.hanning(arr.size)))
    if spectrum.size <= 1:
        return 0.0
    spectrum[0] = 0.0
    peak_index = int(np.argmax(spectrum))
    return float(peak_index * sample_rate / arr.size)


def safe_normalize(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale audio to [-1, 1] while guarding against silence or invalid data."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Audio is empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Audio contains NaN or Inf values.")
    peak = float(np.max(np.abs(arr)))
    if peak < eps:
        return np.zeros_like(arr, dtype=np.float32)
    arr = arr / max(peak, 1.0)
    arr = np.clip(arr, -1.0, 1.0)
    return arr.astype(np.float32)


def preprocess_audio(
    audio: np.ndarray,
    sr: int = AUDIO.SAMPLE_RATE,
    target_sr: int = AUDIO.SAMPLE_RATE,
    window_length: int | None = None,
) -> np.ndarray:
    """Shared audio preprocessing pipeline used by training and inference."""
    if audio is None:
        raise ValueError("Audio input is None.")

    mono = ensure_mono(audio)
    mono = resample_audio(mono, sr, target_sr)
    mono = safe_normalize(mono)

    if window_length is not None:
        if mono.size > window_length:
            mono = mono[:window_length]
        elif mono.size < window_length:
            mono = np.pad(mono, (0, window_length - mono.size), mode="constant")

    return mono.astype(np.float32)
