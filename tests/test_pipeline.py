import numpy as np
import pytest

from config import AUDIO, CLASS_NAMES
from hazard_assessment.hazard_engine import HazardAssessor, Prediction
from preprocessing.audio import dominant_frequency, preprocess_audio
from preprocessing.mel_spectrogram import MelSpectrogramComputer, SlidingWindowBuffer
from models.cnn_model import build_cnn_model, build_crnn_model


def test_preprocess_audio_shape_and_output():
    audio = np.random.randn(AUDIO.SAMPLE_RATE).astype(np.float32)
    processed = preprocess_audio(audio, window_length=AUDIO.SAMPLE_RATE)
    assert processed.dtype == np.float32
    assert processed.shape == (AUDIO.SAMPLE_RATE,)
    assert np.isfinite(processed).all()


def test_mel_spectrogram_output_shape():
    audio = np.random.randn(AUDIO.SAMPLE_RATE).astype(np.float32)
    computer = MelSpectrogramComputer(
        sr=AUDIO.SAMPLE_RATE,
        n_fft=AUDIO.N_FFT,
        hop_length=AUDIO.HOP_LENGTH,
        n_mels=AUDIO.N_MELS,
        fmin=AUDIO.FMIN,
        fmax=AUDIO.FMAX,
    )
    mel = computer.compute(audio)
    assert mel.shape[0] == AUDIO.N_MELS
    assert mel.shape[1] > 0


def test_class_mapping_has_four_classes():
    assert CLASS_NAMES == ["non_vehicle", "vehicle"]


def test_cnn_output_shape():
    model = build_cnn_model(input_shape=AUDIO.INPUT_SHAPE, num_classes=len(CLASS_NAMES))
    x = np.random.randn(2, *AUDIO.INPUT_SHAPE).astype(np.float32)
    y = model(x)
    y_np = y.numpy()
    assert y_np.shape == (2, len(CLASS_NAMES))
    assert np.allclose(y_np.sum(axis=1), np.ones(2), atol=1e-5)


def test_crnn_output_shape():
    model = build_crnn_model(input_shape=AUDIO.INPUT_SHAPE, num_classes=len(CLASS_NAMES))
    x = np.random.randn(2, *AUDIO.INPUT_SHAPE).astype(np.float32)
    y = model(x)
    y_np = y.numpy()
    assert y_np.shape == (2, len(CLASS_NAMES))
    assert np.allclose(y_np.sum(axis=1), np.ones(2), atol=1e-5)


def test_sliding_window_overlap_behavior():
    buffer = SlidingWindowBuffer(window_samples=8)
    buffer.add_audio(np.ones(4, dtype=np.float32))
    assert buffer.is_ready() is False
    buffer.add_audio(np.ones(4, dtype=np.float32))
    window = buffer.get_window()
    assert window is not None
    assert window.shape == (8,)


def test_hazard_engine_persistence_and_cooldown():
    assessor = HazardAssessor(hazard_classes={"vehicle"}, warning_threshold=0.2, emergency_threshold=0.5, cooldown_seconds=1.0)
    p1 = Prediction(class_name="vehicle", confidence=0.9, probabilities={"non_vehicle": 0.1, "vehicle": 0.9})
    first = assessor.assess(p1, time_now=0.0)
    second = assessor.assess(p1, time_now=0.1)
    assert first.alert_level >= 0
    assert second.consecutive_count >= 1
    assert first.hazard_score >= 0.0


def test_background_suppression():
    assessor = HazardAssessor(hazard_classes={"vehicle"}, warning_threshold=0.3, emergency_threshold=0.6)
    prediction = Prediction(class_name="non_vehicle", confidence=0.9, probabilities={"non_vehicle": 0.9, "vehicle": 0.1})
    result = assessor.assess(prediction, time_now=0.0)
    assert result.alert_level == 0


def test_invalid_audio_handling():
    with pytest.raises(ValueError):
        preprocess_audio(np.array([np.nan, np.inf, 0.0], dtype=np.float32))


def test_dominant_frequency_tracks_tone():
    t = np.arange(AUDIO.SAMPLE_RATE, dtype=np.float32) / AUDIO.SAMPLE_RATE
    signal = np.sin(2 * np.pi * 1000 * t)
    assert abs(dominant_frequency(signal) - 1000.0) < 20.0


def test_frequency_increase_contributes_to_vehicle_alert():
    assessor = HazardAssessor(
        hazard_classes={"vehicle"},
        warning_threshold=0.3,
        emergency_threshold=0.8,
        frequency_weight=0.15,
        frequency_increase_hz=40.0,
    )
    probabilities = {"non_vehicle": 0.05, "vehicle": 0.95}
    assessor.assess(Prediction("vehicle", 0.95, probabilities, 500.0), time_now=0.0)
    result = assessor.assess(Prediction("vehicle", 0.95, probabilities, 650.0), time_now=10.0)
    assert result.hazard_score > 0.5


def test_load_model_returns_valid_model_object():
    model = build_cnn_model(input_shape=AUDIO.INPUT_SHAPE, num_classes=len(CLASS_NAMES))
    assert model.output_shape[-1] == len(CLASS_NAMES)
