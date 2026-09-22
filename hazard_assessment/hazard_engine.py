from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


ALERT_LEVELS = {
    0: {"label": "SAFE", "color": "#00d084"},
    1: {"label": "CAUTION", "color": "#f5b700"},
    2: {"label": "VEHICLE SOUND DETECTED", "color": "#ff8c42"},
    3: {"label": "HIGH ACOUSTIC RISK", "color": "#ff3b30"},
}


@dataclass
class Prediction:
    class_name: str
    confidence: float
    probabilities: dict[str, float]
    dominant_frequency_hz: float | None = None
    acoustic_energy: float | None = None


@dataclass
class HazardAssessment:
    hazard_score: float
    alert_level: int
    alert_label: str
    alert_color: str
    confidence: float
    consecutive_count: int
    temporal_stability: float
    window_consistency: float
    cooldown_active: bool
    frequency_rise: float
    energy_rise: float


class HazardAssessor:
    """Simple scientifically safe hazard engine for a lightweight MVP."""

    def __init__(
        self,
        hazard_classes: set[str] | None = None,
        warning_threshold: float = 0.35,
        emergency_threshold: float = 0.65,
        cooldown_seconds: float = 5.0,
        max_consecutive: int = 5,
        history_window: int = 8,
        confidence_weight: float = 0.35,
        consecutive_weight: float = 0.25,
        stability_weight: float = 0.20,
        consistency_weight: float = 0.20,
        frequency_weight: float = 0.15,
        frequency_increase_hz: float = 40.0,
        energy_weight: float = 0.15,
        energy_increase_ratio: float = 0.20,
    ):
        self.hazard_classes = hazard_classes or set()
        self.warning_threshold = warning_threshold
        self.emergency_threshold = emergency_threshold
        self.cooldown_seconds = cooldown_seconds
        self.max_consecutive = max_consecutive
        self.history_window = history_window
        self.confidence_weight = confidence_weight
        self.consecutive_weight = consecutive_weight
        self.stability_weight = stability_weight
        self.consistency_weight = consistency_weight
        self.frequency_weight = frequency_weight
        self.frequency_increase_hz = frequency_increase_hz
        self.energy_weight = energy_weight
        self.energy_increase_ratio = energy_increase_ratio

        self._history: deque[str] = deque(maxlen=self.history_window)
        self._recent_confidences: deque[float] = deque(maxlen=self.history_window)
        self._last_alert_time = 0.0
        self._consecutive_vehicle = 0
        self._last_prediction: str | None = None
        self._frequency_history: deque[float] = deque(maxlen=self.history_window)
        self._energy_history: deque[float] = deque(maxlen=self.history_window)

    def assess(self, prediction: Prediction, time_now: float | None = None) -> HazardAssessment:
        """Assess a single prediction against recent prediction history."""
        if time_now is None:
            import time
            time_now = time.time()

        current_name = prediction.class_name
        self._history.append(current_name)
        self._recent_confidences.append(float(prediction.confidence))
        if prediction.dominant_frequency_hz is not None:
            self._frequency_history.append(float(prediction.dominant_frequency_hz))
        if prediction.acoustic_energy is not None:
            self._energy_history.append(float(prediction.acoustic_energy))

        if current_name in self.hazard_classes:
            self._consecutive_vehicle += 1
        else:
            self._consecutive_vehicle = 0

        recent_vehicle = sum(1 for item in self._history if item in self.hazard_classes)
        vehicle_ratio = recent_vehicle / max(len(self._history), 1)

        if self._last_prediction is None:
            last_same = 1.0
        else:
            last_same = 1.0 if current_name == self._last_prediction else 0.0

        if len(self._history) > 1:
            same_class_count = sum(1 for item in self._history if item == current_name)
            temporal_stability = same_class_count / len(self._history)
        else:
            temporal_stability = 1.0

        confidence_component = clamp(float(prediction.confidence), 0.0, 1.0)
        consecutive_component = clamp(self._consecutive_vehicle / max(self.max_consecutive, 1), 0.0, 1.0)
        stability_component = clamp(temporal_stability, 0.0, 1.0)
        consistency_component = clamp(vehicle_ratio, 0.0, 1.0)
        frequency_component = 0.0
        if len(self._frequency_history) >= 2:
            frequency_delta = self._frequency_history[-1] - self._frequency_history[-2]
            frequency_component = clamp(frequency_delta / max(self.frequency_increase_hz, 1.0))
        energy_component = 0.0
        if len(self._energy_history) >= 2:
            previous_energy = max(self._energy_history[-2], 1e-8)
            energy_ratio = (self._energy_history[-1] - previous_energy) / previous_energy
            energy_component = clamp(energy_ratio / max(self.energy_increase_ratio, 1e-6))

        hazard_score = (
            confidence_component * self.confidence_weight
            + consecutive_component * self.consecutive_weight
            + stability_component * self.stability_weight
            + consistency_component * self.consistency_weight
            + frequency_component * self.frequency_weight
            + energy_component * self.energy_weight
        )

        if current_name not in self.hazard_classes:
            hazard_score *= 0.25

        alert_level = 0
        trend_present = frequency_component > 0.25 or energy_component > 0.25
        if hazard_score >= self.emergency_threshold and trend_present:
            alert_level = 3
        elif hazard_score >= self.warning_threshold:
            alert_level = 2
        elif hazard_score > 0.15:
            alert_level = 1

        cooldown_active = (time_now - self._last_alert_time) < self.cooldown_seconds and alert_level > 0
        if cooldown_active and alert_level > 0:
            alert_level = max(0, alert_level - 1)

        if alert_level > 0:
            self._last_alert_time = time_now

        self._last_prediction = current_name
        alert_label = ALERT_LEVELS.get(alert_level, ALERT_LEVELS[0])["label"]
        alert_color = ALERT_LEVELS.get(alert_level, ALERT_LEVELS[0])["color"]

        return HazardAssessment(
            hazard_score=float(hazard_score),
            alert_level=int(alert_level),
            alert_label=alert_label,
            alert_color=alert_color,
            confidence=confidence_component,
            consecutive_count=self._consecutive_vehicle,
            temporal_stability=float(temporal_stability),
            window_consistency=float(vehicle_ratio),
            cooldown_active=cooldown_active,
            frequency_rise=float(frequency_component),
            energy_rise=float(energy_component),
        )

    def reset(self) -> None:
        self._history.clear()
        self._recent_confidences.clear()
        self._consecutive_vehicle = 0
        self._last_prediction = None
        self._last_alert_time = 0.0
        self._frequency_history.clear()
        self._energy_history.clear()


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(np.clip(value, lower, upper))
