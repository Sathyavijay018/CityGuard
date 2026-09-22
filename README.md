# 🚨 CityGuard — AI-Powered Pedestrian Safety Assistant

**Real-time acoustic hazard detection for pedestrian safety using a CNN-based deep learning pipeline with confidence-based hazard assessment.**

> Built for Final Year Projects, IEEE Publication, Smart India Hackathon, and MSME Innovation.

---

## Architecture Overview

```
┌─────────────┐
│  Microphone │
└──────┬──────┘
       │ Raw audio chunks (0.5s)
       ▼
┌─────────────────────┐
│  Sliding Window      │  2.0s overlapping buffer
│  (preprocessing/)    │  prevents boundary misses
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Audio Preprocessing │  Normalize → Noise Reduction
│  (preprocessing/)    │  → Band-pass Filter
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Mel Spectrogram     │  128 mel bands × 86 frames
│  (preprocessing/)    │  Log-scaled, NumPy STFT
└──────┬──────────────┘
       │  (128, 86, 1)
       ▼
┌─────────────────────┐
│  CNN Classifier      │  Conv2D(32→64→128) + BN
│  (models/)           │  + MaxPool → GAP → Dense
└──────┬──────────────┘
       │  8-class softmax
       ▼
┌─────────────────────┐
│  Hazard Assessment   │  Confidence + Consecutive
│  (hazard_assessment/)│  + Stability + Consistency
└──────┬──────────────┘
       │  hazard_score (0-1)
       ▼
┌─────────────────────┐
│  Three-Level Alert   │  Level 1: Warning (Yellow)
│  System              │  Level 2: Approaching (Orange)
│                      │  Level 3: Emergency (Red)
└─────────────────────┘
```

---

## Sound Classes (8)

| # | Class | Description |
|---|-------|-------------|
| 1 | `car_horn` | Car horn / automobile beep |
| 2 | `bike_horn` | Two-wheeler electronic horn |
| 3 | `truck_horn` | Heavy vehicle air horn |
| 4 | `ambulance_siren` | Ambulance wailing siren |
| 5 | `police_siren` | Police yelp siren |
| 6 | `fire_engine_siren` | Fire truck priority siren |
| 7 | `tire_screech` | Tire skid / brake noise |
| 8 | `background_noise` | Ambient / silence |

---

## Key Innovation — Confidence-Based Hazard Assessment

Instead of triggering alerts from a single prediction, CityGuard computes a **composite hazard score** from four signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| Prediction Confidence | 30% | CNN softmax probability |
| Consecutive Detections | 25% | Hazard predictions in a row |
| Temporal Stability | 25% | Same class across recent predictions |
| Window Consistency | 20% | Fraction of hazard predictions in window |

Alerts fire only when the score exceeds a dynamic threshold, with a cooldown period. This **dramatically reduces false positives** while maintaining responsiveness.

---

## Three-Level Alert System

| Level | Label | Color | Trigger |
|-------|-------|-------|---------|
| 1 | Vehicle Nearby | 🟡 Yellow | Hazard score ≥ 0.35 |
| 2 | Vehicle Approaching | 🟠 Orange | Score ≥ 0.35 + 3 consecutive |
| 3 | Immediate Danger | 🔴 Red | Hazard score ≥ 0.65 |

---

## Installation

### Prerequisites
- Python 3.9+
- Working microphone
- On Linux: `sudo apt-get install libsndfile1 portaudio19-dev`

### Setup

```bash
cd SoundClassifier
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Generate Training Data

```bash
python create_dummy_data.py --samples 100
```

For production, replace the synthetic files in `data/` with real recordings and re-run.

### Train the CNN

```bash
python model_trainer.py --epochs 20
```

This produces:
- `models/cityguard_cnn.keras` — trained Keras model
- `models/metadata.json` — evaluation metrics, class names, training history

### Run the Dashboard

```bash
streamlit run app.py
```

---

## Folder Structure

```
SoundClassifier/
├── app.py                        # Streamlit dashboard (main entry point)
├── audio_processor.py            # Queue-based mic streaming (preserved)
├── config.py                     # Centralised audio/model/hazard config
├── model_trainer.py              # CNN training pipeline
├── create_dummy_data.py          # Synthetic 8-class data generator
├── feature_extraction.py         # Legacy feature extractor (kept for reference)
├── requirements.txt              # Python dependencies
├── style.css                     # Glassmorphism UI theme
│
├── preprocessing/                # Audio preprocessing package
│   ├── __init__.py
│   ├── audio.py                  # Noise reduction, normalization, filtering
│   └── mel_spectrogram.py        # Mel spectrogram + sliding window buffer
│
├── models/                       # Trained model & architecture
│   ├── __init__.py
│   ├── cnn_model.py              # CNN definition & inference helpers
│   ├── cityguard_cnn.keras       # (generated) trained model
│   └── metadata.json             # (generated) evaluation results
│
├── hazard_assessment/            # Core innovation module
│   ├── __init__.py
│   └── hazard_engine.py          # Hazard scoring + three-level alerts
│
├── utils/                        # Utility helpers
│   ├── __init__.py
│   └── visualization.py          # Streamlit HTML rendering
│
├── data/                         # Training audio files (WAV)
│   ├── car_horn/
│   ├── bike_horn/
│   ├── truck_horn/
│   ├── ambulance_siren/
│   ├── police_siren/
│   ├── fire_engine_siren/
│   ├── tire_screech/
│   └── background_noise/
│
└── logs/                         # Runtime logs
```

---

## Evaluation Metrics

After training, the following metrics are available on the **Model Info** page:

- **Accuracy** — overall classification accuracy
- **F1 Score** — weighted F1 across all classes
- **Confusion Matrix** — interactive Plotly heatmap
- **False Positive Rate** — per-class FPR bar chart
- **Training Curves** — loss and accuracy over epochs

---

## Performance Optimisations

| Technique | Benefit |
|-----------|---------|
| Pure NumPy STFT | No librosa dependency at inference time |
| Pre-computed mel filter-bank | O(n_mels × n_freqs) matrix multiply |
| Sliding window buffer | No redundant audio re-processing |
| Global Average Pooling | Fewer parameters than Flatten + Dense |
| Early stopping + LR scheduling | Efficient training convergence |
| Lightweight preprocessing | Skips heavy noise reduction for latency |

---

## Future Enhancements

- [ ] TensorFlow Lite deployment for mobile / edge devices
- [ ] YAMNet fine-tuning as an alternative backbone
- [ ] Real-world dataset integration (Freesound, ESC-50)
- [ ] GPS-based location tagging for alerts
- [ ] Bluetooth haptic feedback for wearable devices
- [ ] Federated learning for crowd-sourced model improvement

---

## License

This project is developed for academic and research purposes.
