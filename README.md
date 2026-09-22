# CityGuard

CityGuard is a lightweight pedestrian safety MVP that listens to environmental audio, extracts log-Mel features, and classifies audio into two classes: vehicle and non-vehicle.

The prototype is intentionally simple and transparent: it prioritizes trustworthy local inference, clear state reporting, and safe alert logic over complex dashboards or cloud integrations.

## Purpose

The target use case is a pedestrian who may be distracted while walking and may miss a nearby vehicle sound. The microphone-based prototype tries to recognize important vehicle sounds from environmental audio and raise a cautious alert only after persistent evidence.

## Architecture

- Audio capture via microphone or uploaded file
- Mono conversion and resampling to 16 kHz
- 2.0 s sliding window with 0.5 s step
- Log-Mel spectrogram using 128 bins and 1024 FFT
- CNN baseline or CNN-GRU model
- Binary softmax classification: `vehicle` / `non_vehicle`
- Acoustic-risk engine using persistence, measured energy change, and dominant-frequency change
- Minimal Streamlit interface

## Binary taxonomy

- `non_vehicle`
- `vehicle`

## Folder structure

- `app.py` — Streamlit demo interface
- `audio_processor.py` — microphone streaming helper
- `config.py` — centralized config
- `create_dummy_data.py` — explicitly synthetic binary development dataset generator
- `model_trainer.py` — training pipeline
- `verify_setup.py` — environment validation
- `preprocessing/` — shared audio and Mel preprocessing
- `models/` — saved model(s)
- `hazard_assessment/` — temporal alert logic
- `tests/` — critical pipeline tests
- `results/` — evaluation outputs from experiments
- `data/` — raw and processed dataset folders

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset preparation

The active MVP uses the supplied binary dataset:

- `non_vehicle/`
- `vehicle/`

These folders may be placed at the project root, as in the current dataset, or under another directory passed with `--data-dir`.

The project also supports a development-only synthetic generator:

```bash
python create_dummy_data.py --samples 20 --out synthetic_data
```

Important rules:

- Do not mix recordings from the same source file across train/test splits.
- Segments from the same original recording must never appear in both training and validation/test sets.
- Augment training data only; do not augment validation or test data.
- Synthetic audio should be treated as a development utility, not real-world validation.

## Training

Training the baseline CNN:

```bash
python model_trainer.py --model cnn --data-dir .
```

Training the CNN-GRU variant:

```bash
python model_trainer.py --model crnn --data-dir .
```

## Evaluation

```bash
python model_trainer.py --model cnn --data-dir .
python model_trainer.py --model crnn --data-dir .
```

The evaluation pipeline saves metrics in `results/` and should report real measured values only.

## Running the MVP

Launch the Streamlit app:

```bash
streamlit run app.py
```

Windows convenience command:

```bat
run_app.bat
```

## Vercel browser demo

The `web/` folder is a static browser MVP for Vercel. It supports microphone capture
and uploaded audio without a Python server. Deploy the repository to Vercel with the
default settings; `vercel.json` routes the site to `web/index.html`.

The browser build performs real client-side audio feature analysis and acoustic-risk
assessment. It does not load the Keras CNN directly because Keras models require
conversion to a browser format or a separate inference API. Use the Streamlit app for
the exact trained CNN inference until that deployment step is added.

Local preview:

```bash
python -m http.server 4173
```

Then open `http://localhost:4173/web/`.

## File analysis mode

The app supports uploading a WAV/MP3/FLAC/OGG file. The same preprocessing and model inference pipeline is used for recorded input, with a simple timeline and spectrogram view.

## Verification

```bash
python verify_setup.py
```

The script checks:

- Python version
- dependency availability
- TensorFlow import
- model presence
- class mapping
- preprocessing
- microphone availability where possible

## Limitations

- Single microphone only
- No physical distance estimation: “HIGH ACOUSTIC RISK” is only a measured-audio proxy, not a distance claim
- No guaranteed approach-direction estimation
- Performance depends on dataset quality and environment
- Overlapping or unusual sounds may affect predictions
- Prototype requires real-world field validation before claiming operational readiness

## Research use

This repository is suitable for comparing a CNN baseline against a lightweight CNN-GRU model under the same binary dataset split. Keep experiments reproducible and report real measured metrics only.
