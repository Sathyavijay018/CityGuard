from __future__ import annotations

import importlib
import os
import sys

from config import AUDIO, CLASS_NAMES, DISPLAY_NAMES
from preprocessing.audio import preprocess_audio
from preprocessing.mel_spectrogram import MelSpectrogramComputer


FAILURES: list[str] = []


def check(cond: bool, message: str) -> None:
    if not cond:
        FAILURES.append(message)


check(sys.version_info[:2] >= (3, 9), "Python 3.9+ is required.")

for module_name in [
    "tensorflow",
    "streamlit",
    "numpy",
    "pandas",
    "scipy",
    "sounddevice",
    "soundfile",
    "sklearn",
    "plotly",
]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover
        FAILURES.append(f"Missing dependency {module_name}: {exc}")

check(CLASS_NAMES == ["non_vehicle", "vehicle"], "Expected binary classes: non_vehicle and vehicle.")
check(set(CLASS_NAMES) == set(DISPLAY_NAMES.keys()), "Class mapping and display labels are inconsistent.")

try:
    import numpy as np

    x = np.random.randn(AUDIO.SAMPLE_RATE).astype(np.float32)
    y = preprocess_audio(x, window_length=AUDIO.SAMPLE_RATE)
    check(y.shape == (AUDIO.SAMPLE_RATE,), "Audio preprocessing must preserve 1-second length for the verification signal.")
    mel = MelSpectrogramComputer(
        sr=AUDIO.SAMPLE_RATE,
        n_fft=AUDIO.N_FFT,
        hop_length=AUDIO.HOP_LENGTH,
        n_mels=AUDIO.N_MELS,
        fmin=AUDIO.FMIN,
        fmax=AUDIO.FMAX,
    ).compute(y)
    check(mel.shape[0] == AUDIO.N_MELS, "Mel spectrogram height must match n_mels.")
    check(mel.shape[1] > 0, "Mel spectrogram time axis must be non-empty.")
except Exception as exc:  # pragma: no cover
    FAILURES.append(f"Preprocessing check failed: {exc}")

try:
    import sounddevice as sd
    check(sd.query_devices() is not None, "Microphone device query did not return a usable result.")
except Exception:  # pragma: no cover
    FAILURES.append("Microphone detection unavailable.")

model_exists = os.path.exists(os.path.join("models", "cityguard_cnn.keras")) or os.path.exists(os.path.join("models", "cityguard_crnn.keras"))
check(model_exists, "No trained model found; run training before the demo.")

if FAILURES:
    print("CITYGUARD SETUP CHECK FAILED")
    for item in FAILURES:
        print(f"- {item}")
    raise SystemExit(1)

print("CITYGUARD READY FOR DEMO")
