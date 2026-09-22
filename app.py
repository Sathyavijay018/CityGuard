from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
from matplotlib import pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from config import AUDIO, CLASS_NAMES, DISPLAY_NAMES, HAZARD
from hazard_assessment.hazard_engine import HazardAssessor, Prediction
from models.cnn_model import load_model, predict_single
from preprocessing.audio import dominant_frequency, preprocess_audio
from preprocessing.mel_spectrogram import MelSpectrogramComputer, SlidingWindowBuffer


@st.cache_resource
def load_cityguard_model():
    candidate_paths = [
        Path("models/cityguard_cnn.keras"),
        Path("models/cityguard_crnn.keras"),
        Path("models/model.keras"),
    ]
    for path in candidate_paths:
        if path.exists():
            model, _ = load_model(path, CLASS_NAMES)
            return model, path.name
    model, _ = load_model(Path("models/cityguard_cnn.keras"), CLASS_NAMES)
    return model, "default CNN"


@st.cache_data
def load_evaluation_summary():
    import json

    summary = {"status": "Not evaluated"}
    for candidate in [
        Path("results/cnn_metrics.json"),
        Path("results/crnn_metrics.json"),
        Path("results/model_comparison.csv"),
    ]:
        if candidate.exists():
            try:
                if candidate.suffix == ".json":
                    summary = json.loads(candidate.read_text())
                    summary["status"] = "Evaluated"
                    return summary
                if candidate.suffix == ".csv":
                    summary = {"status": "Evaluated", "csv": str(candidate)}
                    return summary
            except Exception:
                return {"status": "Not evaluated"}
    return summary


st.set_page_config(page_title="CityGuard", page_icon="🚨", layout="wide")
st.title("CITYGUARD")
st.caption("Real-time acoustic pedestrian safety prototype")
st.info(
    "HIGH ACOUSTIC RISK means persistent vehicle classification plus rising measured "
    "acoustic energy or dominant frequency. It is not a physical distance measurement."
)

if not Path("models").exists():
    st.warning("No models directory found. Train a model first with: python model_trainer.py")

model, model_name = load_cityguard_model()
computer = MelSpectrogramComputer(
    sr=AUDIO.SAMPLE_RATE,
    n_fft=AUDIO.N_FFT,
    hop_length=AUDIO.HOP_LENGTH,
    n_mels=AUDIO.N_MELS,
    fmin=AUDIO.FMIN,
    fmax=AUDIO.FMAX,
)
assessor = HazardAssessor(
    hazard_classes={"vehicle"},
    warning_threshold=0.35,
    emergency_threshold=0.65,
    cooldown_seconds=5.0,
    history_window=8,
    confidence_weight=HAZARD.CONFIDENCE_WEIGHT,
    consecutive_weight=HAZARD.CONSECUTIVE_WEIGHT,
    stability_weight=HAZARD.STABILITY_WEIGHT,
    consistency_weight=HAZARD.CONSISTENCY_WEIGHT,
    frequency_weight=HAZARD.FREQUENCY_WEIGHT,
    frequency_increase_hz=HAZARD.FREQUENCY_INCREASE_HZ,
    energy_weight=HAZARD.ENERGY_WEIGHT,
    energy_increase_ratio=HAZARD.ENERGY_INCREASE_RATIO,
)

if "history" not in st.session_state:
    st.session_state.history = []

left, right = st.columns([2, 1])
with left:
    start_monitoring = st.button("START MONITORING", type="primary")
with right:
    stop_monitoring = st.button("STOP")

status = st.empty()
prob_placeholder = st.empty()

if start_monitoring:
    try:
        import sounddevice as sd

        stream = sd.InputStream(samplerate=AUDIO.SAMPLE_RATE, channels=1, dtype="float32")
        stream.start()
        rolling_buffer = SlidingWindowBuffer(
            int(AUDIO.SAMPLE_RATE * AUDIO.WINDOW_DURATION),
            AUDIO.OVERLAP_RATIO,
        )
        status.markdown(
            "<div style='padding:16px; border-radius:12px; background:#102a22; color:#d4ffe8;'><b>LIVE MICROPHONE</b><br>Listening for vehicle-like acoustic events.</div>",
            unsafe_allow_html=True,
        )

        for _ in range(120):
            if stop_monitoring:
                break
            try:
                block, _ = stream.read(1600)
            except Exception:
                break
            block = np.asarray(block, dtype=np.float32).reshape(-1)
            rolling_buffer.add_audio(block)
            if not rolling_buffer.is_ready():
                continue
            window = rolling_buffer.get_window()
            raw_rms = float(np.sqrt(np.mean(np.square(window))))
            audio = preprocess_audio(
                window,
                sr=AUDIO.SAMPLE_RATE,
                target_sr=AUDIO.SAMPLE_RATE,
                window_length=int(AUDIO.SAMPLE_RATE * AUDIO.WINDOW_DURATION),
            )
            mel = computer.compute(audio)
            pred_name, confidence, probabilities = predict_single(model, mel, CLASS_NAMES)
            frequency_hz = dominant_frequency(audio, AUDIO.SAMPLE_RATE)
            hazard = assessor.assess(
                Prediction(pred_name, confidence, probabilities, frequency_hz, raw_rms),
                time_now=time.time(),
            )

            st.session_state.history.insert(0, {
                "time": time.strftime("%H:%M:%S"),
                "class": DISPLAY_NAMES.get(pred_name, pred_name),
                "confidence": round(confidence * 100, 1),
                "alert": hazard.alert_label,
            })
            st.session_state.history = st.session_state.history[:10]

            status_html = (
                f"<div style='padding:18px; border-radius:14px; background:{hazard.alert_color}; color:white;'>"
                f"<h3>{hazard.alert_label}</h3>"
                f"<p>Detected sound: {DISPLAY_NAMES.get(pred_name, pred_name)}<br>Confidence: {confidence * 100:.1f}%</p>"
                "</div>"
            )
            status.markdown(status_html, unsafe_allow_html=True)

            bars = "".join(
                (
                    f"<div style='margin:4px 0;'><div style='font-size:12px; color:#dfe7f5;'>{DISPLAY_NAMES.get(name, name)}</div>"
                    "<div style='height:10px; width:100%; background:#1d2433; border-radius:6px; overflow:hidden;'>"
                    f"<div style='height:100%; width:{prob * 100:.1f}%; background:#38bdf8; display:block;'></div></div></div>"
                )
                for name, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            )
            prob_placeholder.markdown(
                f"<div style='padding:12px; background:#0c1320; border-radius:12px;'>{bars}</div>",
                unsafe_allow_html=True,
            )
            time.sleep(0.1)
        stream.stop()
        stream.close()
    except Exception as exc:
        st.error(f"Microphone monitoring failed: {exc}")

st.markdown("---")

uploaded = st.file_uploader("Analyze Audio File", type=["wav", "mp3", "flac", "ogg"], accept_multiple_files=False)
if uploaded is not None:
    try:
        audio, sr = sf.read(uploaded)
        audio = np.asarray(audio, dtype=np.float32)
        audio = preprocess_audio(
            audio,
            sr=sr,
            target_sr=AUDIO.SAMPLE_RATE,
            window_length=int(AUDIO.SAMPLE_RATE * AUDIO.WINDOW_DURATION),
        )
        mel = computer.compute(audio)
        pred_name, confidence, probabilities = predict_single(model, mel, CLASS_NAMES)
        st.subheader("Recorded audio inference")
        st.write(f"Predicted class: {DISPLAY_NAMES.get(pred_name, pred_name)}")
        st.write(f"Confidence: {confidence * 100:.1f}%")
        st.bar_chart({DISPLAY_NAMES.get(name, name): prob for name, prob in probabilities.items()})

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
        ax.set_title("Log-Mel Spectrogram")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Mel bins")
        fig.tight_layout()
        st.pyplot(fig)
    except Exception as exc:
        st.error(f"File analysis failed: {exc}")

st.markdown("---")

summary = load_evaluation_summary()
if summary.get("status") == "Evaluated":
    st.subheader("Evaluation summary")
    if "cnn_metrics" in summary:
        st.json(summary["cnn_metrics"])
else:
    st.info("Not evaluated")

if st.session_state.history:
    st.subheader("Recent detections")
    st.dataframe(st.session_state.history)
else:
    st.caption("No recent detections yet.")
