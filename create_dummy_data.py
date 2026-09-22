"""Generate explicitly synthetic binary development data.

Synthetic files are useful for checking the pipeline only. They are not real
world validation and must not be used to report field performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from config import AUDIO, CLASS_NAMES


def _write(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    signal = signal / max(float(np.max(np.abs(signal))), 1e-8)
    sf.write(path, signal.astype(np.float32), sample_rate)


def generate_dataset(
    base_dir: str = "synthetic_data",
    samples_per_class: int = 20,
    duration: float = 2.0,
    sample_rate: int = AUDIO.SAMPLE_RATE,
) -> None:
    rng = np.random.default_rng(42)
    length = int(duration * sample_rate)
    t = np.arange(length, dtype=np.float32) / sample_rate
    for class_name in CLASS_NAMES:
        folder = Path(base_dir) / class_name
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(samples_per_class):
            if class_name == "vehicle":
                frequency = rng.uniform(150.0, 1200.0)
                signal = np.sin(2 * np.pi * frequency * t)
                signal += 0.25 * np.sin(2 * np.pi * frequency * 2.0 * t)
            else:
                signal = rng.normal(0.0, 0.15, length)
            envelope = np.hanning(length) ** 0.25
            _write(folder / f"{class_name}_{index:03d}.wav", signal * envelope, sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic binary development data.")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sr", type=int, default=AUDIO.SAMPLE_RATE)
    parser.add_argument("--out", type=str, default="synthetic_data")
    args = parser.parse_args()
    generate_dataset(args.out, args.samples, args.duration, args.sr)
