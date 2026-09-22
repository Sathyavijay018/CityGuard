from __future__ import annotations

import numpy as np
import tensorflow as tf

from config import AUDIO


class MelSpectrogramComputer:
    """Compute a stable log-Mel spectrogram using TensorFlow signal ops."""

    def __init__(
        self,
        sr: int = AUDIO.SAMPLE_RATE,
        n_fft: int = AUDIO.N_FFT,
        hop_length: int = AUDIO.HOP_LENGTH,
        n_mels: int = AUDIO.N_MELS,
        fmin: float = AUDIO.FMIN,
        fmax: float = AUDIO.FMAX,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self._mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.n_fft // 2 + 1,
            sample_rate=self.sr,
            lower_edge_hertz=self.fmin,
            upper_edge_hertz=self.fmax,
        )

    def compute(self, audio: np.ndarray) -> np.ndarray:
        """Return a log-Mel spectrogram with shape (n_mels, time_frames)."""
        if audio is None:
            raise ValueError("Audio input is None.")
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            raise ValueError("Audio is empty.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Audio contains NaN or Inf values.")

        if arr.size < self.n_fft:
            arr = np.pad(arr, (0, self.n_fft - arr.size), mode="constant")

        spectrogram = tf.signal.stft(
            tf.convert_to_tensor(arr),
            frame_length=self.n_fft,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
        )
        power = tf.square(tf.abs(spectrogram))
        power = tf.maximum(power, 1e-10)

        mel = tf.matmul(
            tf.transpose(tf.cast(self._mel_matrix, tf.float32)),
            tf.transpose(power),
        )
        mel = tf.math.log(mel + 1e-8)
        mel = tf.cast(mel, tf.float32)
        return mel.numpy()


class SlidingWindowBuffer:
    """Maintain a rolling 2-second audio window for live inference."""

    def __init__(self, window_samples: int, overlap_ratio: float = 0.5):
        self.window_samples = max(1, int(window_samples))
        self.overlap_ratio = max(0.0, min(overlap_ratio, 0.95))
        self.buffer = np.array([], dtype=np.float32)

    def add_audio(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return
        self.buffer = np.concatenate([self.buffer, chunk])
        if self.buffer.size > self.window_samples:
            self.buffer = self.buffer[-self.window_samples:]

    def is_ready(self) -> bool:
        return self.buffer.size >= self.window_samples

    def get_window(self) -> np.ndarray | None:
        if not self.is_ready():
            return None
        window = self.buffer[-self.window_samples:]
        return window.astype(np.float32)

    def reset(self) -> None:
        self.buffer = np.array([], dtype=np.float32)
