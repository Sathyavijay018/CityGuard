"""Generate synthetic training data for all 8 CityGuard sound classes.

Each class produces ``samples_per_class`` synthetic WAV files that approximate
the acoustic characteristics of the real-world sound.  Replace these with
real recordings for production use.

Usage
-----
    python create_dummy_data.py
    python create_dummy_data.py --samples 200 --sr 22050
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import scipy.signal
import soundfile as sf


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _t(duration: float, sr: int) -> np.ndarray:
    return np.linspace(0, duration, int(sr * duration), endpoint=False)


def _normalize(sig: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(sig))
    return sig / (peak + 1e-8)


def _save(sig: np.ndarray, path: str, sr: int) -> None:
    sig = _normalize(sig) * np.random.uniform(0.6, 0.95)
    sf.write(path, sig.astype(np.float32), sr)


# ------------------------------------------------------------------ #
# Sound generators
# ------------------------------------------------------------------ #

def gen_car_horn(duration: float, sr: int) -> np.ndarray:
    """Mid-frequency harmonic tone with slight vibrato (~350-500 Hz)."""
    t = _t(duration, sr)
    f0 = np.random.uniform(350, 500)
    vibrato = 5 * np.sin(2 * np.pi * 5 * t)
    sig = (
        0.6 * np.sin(2 * np.pi * (f0 + vibrato) * t)
        + 0.3 * np.sin(2 * np.pi * (f0 * 2) * t)
        + 0.15 * np.sin(2 * np.pi * (f0 * 3) * t)
    )
    env = scipy.signal.windows.hann(len(t)) ** 0.5
    return sig * env


def gen_bike_horn(duration: float, sr: int) -> np.ndarray:
    """Higher-pitched electronic horn (~800-1200 Hz)."""
    t = _t(duration, sr)
    f0 = np.random.uniform(800, 1200)
    sig = (
        0.7 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * scipy.signal.square(2 * np.pi * f0 * t, duty=0.5)
    )
    env = scipy.signal.windows.hann(len(t)) ** 0.6
    return sig * env


def gen_truck_horn(duration: float, sr: int) -> np.ndarray:
    """Low-frequency dual-tone air horn (~150-250 Hz)."""
    t = _t(duration, sr)
    f1 = np.random.uniform(150, 200)
    f2 = f1 * 1.5
    sig = (
        0.6 * np.sin(2 * np.pi * f1 * t)
        + 0.5 * np.sin(2 * np.pi * f2 * t)
        + 0.2 * scipy.signal.sawtooth(2 * np.pi * f1 * t)
    )
    # Low-pass to give a bass-heavy character
    b, a = scipy.signal.butter(4, 500 / (sr / 2), "low")
    sig = scipy.signal.filtfilt(b, a, sig)
    env = scipy.signal.windows.hann(len(t)) ** 0.4
    return sig * env


def gen_ambulance_siren(duration: float, sr: int) -> np.ndarray:
    """Slow wailing sweep 600 → 1200 Hz over 2-3 s."""
    t = _t(duration, sr)
    sweep_rate = np.random.uniform(0.4, 0.6)
    sweep = 900 + 300 * np.sin(2 * np.pi * sweep_rate * t)
    phase = 2 * np.pi * np.cumsum(sweep) / sr
    sig = 0.8 * np.sin(phase) + 0.15 * np.sin(3 * phase)
    return sig


def gen_police_siren(duration: float, sr: int) -> np.ndarray:
    """Fast alternating yelp 800 ↔ 1600 Hz."""
    t = _t(duration, sr)
    sweep_rate = np.random.uniform(3.0, 5.0)
    sweep = 1200 + 400 * np.sin(2 * np.pi * sweep_rate * t)
    phase = 2 * np.pi * np.cumsum(sweep) / sr
    sig = 0.7 * np.sin(phase) + 0.2 * np.sin(2 * phase)
    return sig


def gen_fire_engine_siren(duration: float, sr: int) -> np.ndarray:
    """Very slow, deep sweep 400 → 900 Hz."""
    t = _t(duration, sr)
    sweep_rate = np.random.uniform(0.2, 0.35)
    sweep = 650 + 250 * np.sin(2 * np.pi * sweep_rate * t)
    phase = 2 * np.pi * np.cumsum(sweep) / sr
    sig = (
        0.7 * np.sin(phase)
        + 0.3 * scipy.signal.sawtooth(phase)
    )
    b, a = scipy.signal.butter(3, 1500 / (sr / 2), "low")
    sig = scipy.signal.filtfilt(b, a, sig)
    return sig


def gen_tire_screech(duration: float, sr: int) -> np.ndarray:
    """Broadband high-frequency noise burst (2-8 kHz)."""
    t = _t(duration, sr)
    noise = np.random.randn(len(t))
    low = np.random.uniform(2000, 3000)
    high = np.random.uniform(6000, 8000)
    b, a = scipy.signal.butter(3, [low / (sr / 2), high / (sr / 2)], "bandpass")
    sig = scipy.signal.filtfilt(b, a, noise)
    # Add a rising chirp for the "screech" character
    chirp = 0.3 * scipy.signal.chirp(t, f0=2000, f1=7000, t1=duration)
    sig = sig + chirp
    n = len(t)
    n_attack = int(round(0.1 * n))
    n_sustain = int(round(0.7 * n))
    n_decay = n - n_attack - n_sustain
    env = np.concatenate([
        np.linspace(0, 1, n_attack),
        np.ones(n_sustain),
        np.linspace(1, 0, n_decay),
    ])
    return sig * env


def gen_background_noise(duration: float, sr: int) -> np.ndarray:
    """Quiet room ambience / environmental noise."""
    t = _t(duration, sr)
    noise = np.random.randn(len(t))
    b, a = scipy.signal.butter(2, 500 / (sr / 2), "high")
    hiss = scipy.signal.filtfilt(b, a, noise)
    # Add faint low-frequency rumble
    rumble = 0.05 * np.sin(2 * np.pi * np.random.uniform(40, 80) * t)
    sig = 0.003 * hiss + rumble
    return sig


# ------------------------------------------------------------------ #
# Generator map
# ------------------------------------------------------------------ #

GENERATORS: dict[str, callable] = {
    "car_horn": gen_car_horn,
    "bike_horn": gen_bike_horn,
    "truck_horn": gen_truck_horn,
    "ambulance_siren": gen_ambulance_siren,
    "police_siren": gen_police_siren,
    "fire_engine_siren": gen_fire_engine_siren,
    "tire_screech": gen_tire_screech,
    "background_noise": gen_background_noise,
}


# ------------------------------------------------------------------ #
# Dataset creation
# ------------------------------------------------------------------ #

def generate_dataset(
    base_dir: str = "data",
    samples_per_class: int = 100,
    sr: int = 22050,
    duration: float = 2.0,
) -> None:
    """Generate synthetic WAV files for every class."""
    for class_name, gen_fn in GENERATORS.items():
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"Generating {samples_per_class} samples for [{class_name}] ...")
        for i in range(samples_per_class):
            sig = gen_fn(duration, sr)
            _save(sig, os.path.join(class_dir, f"{class_name}_{i:03d}.wav"), sr)
    print(f"\nDataset generated in '{base_dir}/' with {len(GENERATORS)} classes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CityGuard training data")
    parser.add_argument("--samples", type=int, default=100, help="Samples per class")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration (s)")
    parser.add_argument("--out", type=str, default="data", help="Output directory")
    args = parser.parse_args()
    generate_dataset(base_dir=args.out, samples_per_class=args.samples, sr=args.sr, duration=args.duration)
