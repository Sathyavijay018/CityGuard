# 🚨 CitySound Guard: Intelligent Sound Classification System

An intelligent sound classification system built with Python and Streamlit that listens to environmental audio and classifies it as either Vehicle Sound or Non-Vehicle Sound in real-time. Features a beautiful modern dashboard UI suitable for smart city monitoring.

## Features

- **Real-time Monitoring**: Uses `sounddevice` to capture microphone input continuously and classifies it every second.
- **Glassmorphism UI**: Dark-themed, frosted glass interface with smooth CSS animations.
- **Intelligent Engine**: Powered by a scikit-learn `RandomForestClassifier`.
- **Advanced Audio Processing**: Extracts 42 features including MFCCs, Spectral Centroid, and Zero Crossing Rate via `librosa`.
- **Alert System**: Employs an intelligent cooldown (5s) and consecutive detection rules (3 in a row) before warning.
- **File Upload & Log**: Test pre-recorded `.wav` files and view/export a session history CSV log.

## Getting Started

### 1. Prerequisites

Ensure you have Python 3.9+ installed and a working microphone connected. You will also need standard system dependencies for audio processing.

On Linux:
```bash
sudo apt-get install libsndfile1 portaudio19-dev python3.12-venv
```

### 2. Installation

Clone/extract the project, then set up the environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Model Training

The application requires a pre-trained model to start. A small dummy dataset generator is included to help you verify everything works end-to-end.

Generate dummy sounds and train the Random Forest:
```bash
python create_dummy_data.py
python model_trainer.py
```

*Note: For production, replace the dummy audio files in `data/vehicle` and `data/non_vehicle` with the actual dataset collected from freesound.org and re-run `python model_trainer.py`.*

### 4. Running the Dashboard

Once the `models/sound_classifier.pkl` and `models/scaler.pkl` are generated, launch the dashboard:

```bash
streamlit run app.py
```

### Application Structure

- `app.py`: Main Streamlit app containing UI and routing.
- `audio_processor.py`: Background microphone listener logic using `sounddevice`.
- `feature_extraction.py`: Performs standard scaler normalization & `librosa` feature engineering.
- `model_trainer.py`: Scrapes the `data/` folder to fit the Random Forest model and saves metadata.
- `style.css`: All the theming overrides to transform Streamlit into a Glassmorphism web app.
- `create_dummy_data.py`: Prepares minimum working `.wav` files for the initial test.

## Notes on "sounddevice"
Since `sounddevice` requests Direct hardware access to the microphone, ensure your browser/OS grants the Python process microphone permissions. If you see silence logs or an ALSA error on Linux, try testing your microphone inputs manually.
