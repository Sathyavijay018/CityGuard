"""CityGuard — Real-Time Pedestrian Safety Assistant
====================================================
Upgraded Streamlit dashboard with CNN-based audio classification,
confidence-based hazard assessment, and three-level alert system.
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime

# Suppress TF info-level stderr noise on Windows
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ------------------------------------------------------------------ #
# Local imports
# ------------------------------------------------------------------ #
from config import AUDIO, HAZARD, DISPLAY_NAMES, HAZARD_CLASSES
from preprocessing.mel_spectrogram import MelSpectrogramComputer, SlidingWindowBuffer
from preprocessing.audio import normalize_audio
from hazard_assessment.hazard_engine import HazardAssessor, Prediction, ALERT_LEVELS
from utils.visualization import format_status_card, format_alert_card, get_risk_color
from models.cnn_model import predict_single

# ------------------------------------------------------------------ #
# Streamlit Page Config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="CityGuard — Pedestrian Safety AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ------------------------------------------------------------------ #
# Load Model (cached)
# ------------------------------------------------------------------ #

@st.cache_resource
def load_model():
    """Load the Keras CNN model and class names."""
    from models.cnn_model import load_trained_model
    model, class_names = load_trained_model("models")
    return model, class_names

model, class_names = load_model()

# ------------------------------------------------------------------ #
# Session State Initialisation
# ------------------------------------------------------------------ #

_DEFAULTS = {
    "log": [],
    "stats": {"total": 0, "hazard": 0, "safe": 0},
    "session_start": None,
    "alert_count": 0,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default if not callable(default) else default()


def add_log(class_name: str, confidence: float, hazard_score: float, alert_level: int):
    """Append a detection record to the session log."""
    st.session_state.stats["total"] += 1
    if class_name in HAZARD_CLASSES:
        st.session_state.stats["hazard"] += 1
    else:
        st.session_state.stats["safe"] += 1

    st.session_state.log.insert(0, {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Sound": DISPLAY_NAMES.get(class_name, class_name),
        "Confidence (%)": round(confidence * 100, 2),
        "Hazard Score": round(hazard_score, 3),
        "Alert": ALERT_LEVELS[alert_level]["label"],
    })
    st.session_state.log = st.session_state.log[:200]


# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #

st.sidebar.markdown(
    "<h2 style='text-align:center; color:#00f0ff;'>🚨 CityGuard AI</h2>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<p style='text-align:center; color:#8892b0;'>Pedestrian Safety Assistant v3.0</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", ["Dashboard", "Detection Log", "Model Info"])

st.sidebar.markdown("---")
sensitivity = st.sidebar.select_slider(
    "Alert Sensitivity",
    options=["Low", "Medium", "High"],
    value="Medium",
)

# Map sensitivity to hazard thresholds
_SENS = {
    "Low":    (0.50, 0.80),
    "Medium": (0.35, 0.65),
    "High":   (0.25, 0.50),
}
warn_t, emerg_t = _SENS[sensitivity]


# ================================================================== #
#  DASHBOARD PAGE
# ================================================================== #

if page == "Dashboard":
    st.markdown(
        "<h1 style='text-align:center; color:#00f0ff;'>Real-Time Monitoring Dashboard</h1>",
        unsafe_allow_html=True,
    )

    if model is None:
        st.error("CNN model not found. Please train the model first:\n\n"
                 "`python create_dummy_data.py && python model_trainer.py`")
        st.stop()

    st.markdown(
        "<p style='text-align:center; color:#00ff99; font-weight:bold;'>"
        "✅ CNN Model Loaded — 8-Class Acoustic Classifier</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_mic = st.checkbox("🎙️ Enable Live Microphone (Press to Start/Stop)")

    placeholder = st.empty()

    if run_mic:
        from audio_processor import AudioListener

        # Initialise pipeline components
        listener = AudioListener(
            sample_rate=AUDIO.SAMPLE_RATE,
            chunk_duration=AUDIO.CHUNK_DURATION,
        )
        window_size = int(AUDIO.SAMPLE_RATE * AUDIO.WINDOW_DURATION)
        buffer = SlidingWindowBuffer(window_size, AUDIO.OVERLAP_RATIO)
        mel_computer = MelSpectrogramComputer(
            sr=AUDIO.SAMPLE_RATE,
            n_fft=AUDIO.N_FFT,
            hop_length=AUDIO.HOP_LENGTH,
            n_mels=AUDIO.N_MELS,
            fmin=AUDIO.FMIN,
            fmax=AUDIO.FMAX,
        )
        assessor = HazardAssessor(
            weight_confidence=HAZARD.WEIGHT_CONFIDENCE,
            weight_consecutive=HAZARD.WEIGHT_CONSECUTIVE,
            weight_stability=HAZARD.WEIGHT_STABILITY,
            weight_consistency=HAZARD.WEIGHT_CONSISTENCY,
            threshold_warning=warn_t,
            threshold_emergency=emerg_t,
            cooldown_seconds=HAZARD.COOLDOWN_SECONDS,
            stability_window=HAZARD.STABILITY_WINDOW,
            max_consecutive=HAZARD.MAX_CONSECUTIVE,
            hazard_classes=HAZARD_CLASSES,
        )

        if st.session_state.session_start is None:
            st.session_state.session_start = time.time()

        listener.start()

        while run_mic:
            chunk = listener.get_latest_chunk()

            if chunk is None:
                time.sleep(0.3)
                continue

            # RMS silence check on the incoming chunk
            rms = np.sqrt(np.mean(chunk ** 2))
            if rms < HAZARD.SILENCE_RMS_THRESHOLD:
                with placeholder.container():
                    st.markdown(
                        "<div class='glass-card' style='text-align:center;'>"
                        "<h3 style='color:#8892b0;'>🎧 Listening… (silence)</h3></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<div class='waveform-container'>"
                        + "".join(
                            "<div class='bar' style='animation:none; height:8%;'></div>"
                            for _ in range(8)
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                time.sleep(0.4)
                continue

            # Feed audio into the sliding-window buffer
            buffer.add_audio(chunk)

            if not buffer.is_ready():
                time.sleep(0.2)
                continue

            window_audio = buffer.get_window()
            if window_audio is None:
                time.sleep(0.2)
                continue

            # --- Inference pipeline ---
            try:
                # Preprocess (lightweight — skip heavy noise reduction for latency)
                audio_clean = normalize_audio(window_audio)

                # Mel spectrogram
                mel = mel_computer.compute(audio_clean)
                # Pad / trim to target width
                T = mel.shape[1]
                if T >= AUDIO.MEL_WIDTH:
                    mel = mel[:, : AUDIO.MEL_WIDTH]
                else:
                    mel = np.pad(mel, ((0, 0), (0, AUDIO.MEL_WIDTH - T)))

                # CNN inference
                pred_class, confidence, all_probs = predict_single(model, mel, class_names)

                # Hazard assessment
                prediction = Prediction(pred_class, confidence, all_probs)
                assessment = assessor.assess(prediction)

                # Fire alert if level > 0
                if assessment.alert_level > 0:
                    assessor.fire_alert()
                    st.session_state.alert_count += 1

                add_log(pred_class, confidence, assessment.hazard_score, assessment.alert_level)

                # --- Render UI ---
                display_name = DISPLAY_NAMES.get(pred_class, pred_class)
                elapsed = time.time() - st.session_state.session_start
                mins, secs = divmod(int(elapsed), 60)

                with placeholder.container():
                    # Mic status
                    st.markdown(
                        f"<div class='mic-status'>"
                        f"<div class='mic-indicator'></div>"
                        f"Live monitoring — RMS {rms:.4f} — {mins:02d}:{secs:02d}</div>",
                        unsafe_allow_html=True,
                    )

                    # Alert banner (if active)
                    alert_html = format_alert_card(
                        assessment.alert_level,
                        assessment.alert_label,
                        assessment.alert_color,
                    )
                    if alert_html:
                        st.markdown(alert_html, unsafe_allow_html=True)

                    # Status card
                    st.markdown(
                        format_status_card(
                            predicted_class=pred_class,
                            display_name=display_name,
                            confidence=assessment.confidence,
                            hazard_score=assessment.hazard_score,
                            alert_label=assessment.alert_label,
                            alert_color=assessment.alert_color,
                            consecutive=assessment.consecutive_count,
                            stability=assessment.temporal_stability,
                            consistency=assessment.window_consistency,
                            cooldown=assessment.cooldown_active,
                        ),
                        unsafe_allow_html=True,
                    )

                    # Animated waveform bars
                    st.markdown(
                        "<div class='waveform-container'>"
                        + "".join("<div class='bar'></div>" for _ in range(8))
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                    # Top-N class probabilities
                    top_n = sorted(all_probs.items(), key=lambda x: -x[1])[:5]
                    prob_cols = st.columns(5)
                    for i, (cls, prob) in enumerate(top_n):
                        label = DISPLAY_NAMES.get(cls, cls)
                        color = get_risk_color(prob)
                        with prob_cols[i]:
                            st.markdown(
                                f"<div style='text-align:center; padding:8px; "
                                f"background:rgba(255,255,255,0.03); border-radius:12px;'>"
                                f"<div style='font-size:0.75rem; color:#8892b0;'>{label}</div>"
                                f"<div style='font-size:1.3rem; font-weight:700; "
                                f"color:{color};'>{prob*100:.1f}%</div></div>",
                                unsafe_allow_html=True,
                            )

            except Exception as e:
                st.error(f"Inference error: {e}")

            time.sleep(0.1)

        listener.stop()
        buffer.reset()


# ================================================================== #
#  DETECTION LOG PAGE
# ================================================================== #

elif page == "Detection Log":
    st.markdown("<h1 style='color:#00f0ff;'>Session Detection Log</h1>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Detections", st.session_state.stats["total"])
    c2.metric("Hazard Sounds", st.session_state.stats["hazard"])
    c3.metric("Safe Sounds", st.session_state.stats["safe"])
    c4.metric("Alerts Fired", st.session_state.alert_count)
    if st.session_state.session_start:
        elapsed = time.time() - st.session_state.session_start
        m, s = divmod(int(elapsed), 60)
        c5.metric("Session Duration", f"{m:02d}:{s:02d}")
    else:
        c5.metric("Session Duration", "—")

    st.markdown("---")

    if st.session_state.log:
        df = pd.DataFrame(st.session_state.log)

        def _highlight_hazard(s):
            return [
                "background-color: rgba(255, 51, 102, 0.15); color: #ff3366"
                if v != "Background Noise" else
                "background-color: rgba(0, 255, 153, 0.15); color: #00ff99"
                for v in s
            ]

        st.dataframe(
            df.style.apply(_highlight_hazard, subset=["Sound"]),
            use_container_width=True,
            height=500,
        )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "cityguard_log.csv", "text/csv")
    else:
        st.info("No detections yet. Start the microphone on the Dashboard.")


# ================================================================== #
#  MODEL INFO PAGE
# ================================================================== #

elif page == "Model Info":
    st.markdown("<h1 style='color:#00f0ff;'>Model & Pipeline Information</h1>", unsafe_allow_html=True)

    st.markdown(
        "<div class='glass-card'>"
        "<h3>CNN Architecture</h3>"
        "<p>3-block convolutional network with Batch Normalisation and Global Average Pooling.</p>"
        "<ul>"
        "<li><b>Input:</b> 128×86 Log-Mel Spectrogram (single channel)</li>"
        "<li><b>Block 1:</b> Conv2D(32) → BatchNorm → MaxPool</li>"
        "<li><b>Block 2:</b> Conv2D(64) → BatchNorm → MaxPool</li>"
        "<li><b>Block 3:</b> Conv2D(128) → BatchNorm → MaxPool</li>"
        "<li><b>Head:</b> GlobalAvgPool → Dense(128) → Dropout(0.5) → Softmax</li>"
        "<li><b>Classes:</b> 8 (Car Horn, Bike Horn, Truck Horn, Ambulance Siren, "
        "Police Siren, Fire Engine Siren, Tire Screech, Background Noise)</li>"
        "</ul>"
        "<h3>Innovation — Confidence-Based Hazard Assessment</h3>"
        "<p>Instead of triggering alerts on a single prediction, CityGuard computes a "
        "composite <b>hazard score</b> from four signals: prediction confidence, consecutive "
        "detections, temporal stability, and sliding-window consistency. Alerts fire only when "
        "the score exceeds a dynamic threshold, with a cooldown period to prevent spam.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Training metrics from metadata.json
    import json
    meta_path = os.path.join("models", "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

        metrics = meta.get("metrics", {})
        st.subheader("Evaluation Metrics")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
        mc2.metric("F1 Score (weighted)", f"{metrics.get('f1_weighted', 0)*100:.1f}%")
        mc3.metric("Classes", meta.get("num_classes", 0))
        mc4.metric("Input Shape", str(meta.get("input_shape", "")))

        # Confusion matrix
        cm = metrics.get("confusion_matrix")
        if cm:
            st.subheader("Confusion Matrix")
            import plotly.graph_objs as go
            cls = meta.get("class_names", [])
            display_cls = [DISPLAY_NAMES.get(c, c) for c in cls]
            fig = go.Figure(data=go.Heatmap(
                z=cm, x=display_cls, y=display_cls,
                colorscale="Blues", text=cm, texttemplate="%{text}",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Training curves
        hist = meta.get("history", {})
        if hist.get("loss"):
            st.subheader("Training Curves")
            tc1, tc2 = st.columns(2)
            with tc1:
                st.line_chart(pd.DataFrame({
                    "Train Loss": hist["loss"],
                    "Val Loss": hist.get("val_loss", []),
                }))
            with tc2:
                st.line_chart(pd.DataFrame({
                    "Train Accuracy": hist["accuracy"],
                    "Val Accuracy": hist.get("val_accuracy", []),
                }))

        # False positive rate
        fpr = metrics.get("false_positive_rate")
        if fpr:
            st.subheader("False Positive Rate per Class")
            fpr_df = pd.DataFrame(
                [{"Class": DISPLAY_NAMES.get(k, k), "FPR": v} for k, v in fpr.items()]
            )
            st.bar_chart(fpr_df.set_index("Class"))
    else:
        st.info("No training metadata found. Train the model to see evaluation results.")
