"""Centralized configuration for the CityGuard MVP."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    """Shared audio acquisition and preprocessing settings."""

    SAMPLE_RATE: int = 16000
    CHUNK_DURATION: float = 0.5
    WINDOW_DURATION: float = 2.0
    STEP_DURATION: float = 0.5
    OVERLAP_RATIO: float = 0.5
    N_FFT: int = 1024
    HOP_LENGTH: int = 256
    N_MELS: int = 128
    FMIN: float = 50.0
    FMAX: float = 8000.0
    MEL_HEIGHT: int = 128
    MEL_WIDTH: int = 125
    INPUT_SHAPE: tuple[int, int, int] = (128, 125, 1)


@dataclass(frozen=True)
class HazardConfig:
    """Scientifically safe hazard thresholds and alert tuning."""

    WARNING_THRESHOLD: float = 0.35
    EMERGENCY_THRESHOLD: float = 0.65
    COOLDOWN_SECONDS: float = 5.0
    STABILITY_WINDOW: int = 8
    MAX_CONSECUTIVE: int = 5
    SILENCE_RMS_THRESHOLD: float = 0.02
    CONFIDENCE_WEIGHT: float = 0.30
    CONSECUTIVE_WEIGHT: float = 0.25
    STABILITY_WEIGHT: float = 0.20
    CONSISTENCY_WEIGHT: float = 0.10
    FREQUENCY_WEIGHT: float = 0.10
    FREQUENCY_INCREASE_HZ: float = 40.0
    ENERGY_WEIGHT: float = 0.15
    ENERGY_INCREASE_RATIO: float = 0.20


@dataclass(frozen=True)
class ModelConfig:
    """Model training settings."""

    BATCH_SIZE: int = 64
    EPOCHS: int = 20
    LEARNING_RATE: float = 0.001
    VALIDATION_SPLIT: float = 0.2
    AUGMENTATION_FACTOR: int = 0
    RANDOM_SEED: int = 42


CLASS_NAMES: list[str] = ["non_vehicle", "vehicle"]
DISPLAY_NAMES: dict[str, str] = {
    "non_vehicle": "Non-vehicle",
    "vehicle": "Vehicle",
}
HAZARD_CLASSES: set[str] = {"vehicle"}

AUDIO = AudioConfig()
HAZARD = HazardConfig()
MODEL = ModelConfig()
