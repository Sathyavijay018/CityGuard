"""Centralized configuration for CityGuard audio classification pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    """Audio acquisition and processing parameters."""
    SAMPLE_RATE: int = 22050
    CHUNK_DURATION: float = 0.5          # seconds per mic chunk
    WINDOW_DURATION: float = 2.0         # sliding window length in seconds
    OVERLAP_RATIO: float = 0.5           # 50% overlap between windows
    N_FFT: int = 2048
    HOP_LENGTH: int = 512
    N_MELS: int = 128
    FMIN: float = 20.0
    FMAX: float = 11025.0
    MEL_HEIGHT: int = 128
    MEL_WIDTH: int = 86                  # ~2s of audio at HOP_LENGTH=512
    INPUT_SHAPE: tuple = (128, 86, 1)


@dataclass(frozen=True)
class HazardConfig:
    """Hazard assessment thresholds and weights."""
    # Signal weights (must sum to ~1.0 for interpretability)
    WEIGHT_CONFIDENCE: float = 0.30
    WEIGHT_CONSECUTIVE: float = 0.25
    WEIGHT_STABILITY: float = 0.25
    WEIGHT_CONSISTENCY: float = 0.20

    # Alert thresholds
    THRESHOLD_WARNING: float = 0.35
    THRESHOLD_EMERGENCY: float = 0.65

    # Cooldown
    COOLDOWN_SECONDS: float = 5.0

    # Sliding window for stability measurement
    STABILITY_WINDOW: int = 5

    # Consecutive detection cap (normalises the signal)
    MAX_CONSECUTIVE: int = 5

    # Silence detection
    SILENCE_RMS_THRESHOLD: float = 0.02


@dataclass(frozen=True)
class ModelConfig:
    """CNN training hyper-parameters."""
    BATCH_SIZE: int = 32
    EPOCHS: int = 20
    LEARNING_RATE: float = 0.001
    VALIDATION_SPLIT: float = 0.2
    AUGMENTATION_FACTOR: int = 3


# 8-class taxonomy for the upgraded pipeline
CLASS_NAMES: list[str] = [
    "car_horn",
    "bike_horn",
    "truck_horn",
    "ambulance_siren",
    "police_siren",
    "fire_engine_siren",
    "tire_screech",
    "background_noise",
]

# Human-readable labels
DISPLAY_NAMES: dict[str, str] = {
    "car_horn": "Car Horn",
    "bike_horn": "Bike Horn",
    "truck_horn": "Truck Horn",
    "ambulance_siren": "Ambulance Siren",
    "police_siren": "Police Siren",
    "fire_engine_siren": "Fire Engine Siren",
    "tire_screech": "Tire Screech",
    "background_noise": "Background Noise",
}

# Which classes are considered hazardous (triggers alerts)
HAZARD_CLASSES: set[str] = {
    "car_horn", "bike_horn", "truck_horn",
    "ambulance_siren", "police_siren", "fire_engine_siren",
    "tire_screech",
}

# Instantiated singletons for convenience
AUDIO = AudioConfig()
HAZARD = HazardConfig()
MODEL = ModelConfig()
