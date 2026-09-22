"""Train the CityGuard CNN on mel-spectrogram data.

Usage
-----
    python model_trainer.py                # train with default settings
    python model_trainer.py --epochs 30    # override hyper-parameters

The script auto-discovers every sub-folder under ``data/`` and treats each
folder name as a class label.  It produces:

* ``models/cityguard_cnn.keras``  — trained Keras model
* ``models/metadata.json``         — class names, metrics, training history
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ #
# Ensure local package imports work when run as a script
# ------------------------------------------------------------------ #
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AUDIO, MODEL, CLASS_NAMES
from preprocessing.mel_spectrogram import MelSpectrogramComputer
from preprocessing.audio import preprocess_audio
from models.cnn_model import build_cnn_model

# Suppress TF info-level stderr noise on Windows
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #

def load_dataset(
    base_dir: str = "data",
    target_width: int = AUDIO.MEL_WIDTH,
    augment: bool = True,
    augment_factor: int = MODEL.AUGMENTATION_FACTOR,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load audio files, compute mel spectrograms, and optionally augment.

    Returns
    -------
    X : ndarray, shape ``(N, n_mels, T, 1)``
    y : ndarray of integer labels
    class_names : list of class label strings (sorted alphabetically)
    """
    import soundfile as sf

    computer = MelSpectrogramComputer(
        sr=AUDIO.SAMPLE_RATE,
        n_fft=AUDIO.N_FFT,
        hop_length=AUDIO.HOP_LENGTH,
        n_mels=AUDIO.N_MELS,
        fmin=AUDIO.FMIN,
        fmax=AUDIO.FMAX,
    )

    # Discover classes — only use folders that match CLASS_NAMES
    all_dirs = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])
    class_dirs = sorted([d for d in all_dirs if d in CLASS_NAMES])
    ignored = [d for d in all_dirs if d not in CLASS_NAMES]
    if ignored:
        print(f"Ignoring legacy folders: {ignored}")
    if not class_dirs:
        raise FileNotFoundError(f"No valid class sub-folders found in '{base_dir}/'")

    print(f"Discovered {len(class_dirs)} classes: {class_dirs}")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for label_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(base_dir, class_name)
        wav_files = sorted(Path(class_path).glob("*.wav"))
        print(f"  [{class_name}] {len(wav_files)} files")

        for fpath in wav_files:
            try:
                audio, sr = sf.read(str(fpath), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != AUDIO.SAMPLE_RATE:
                    # Simple resample via interpolation
                    ratio = AUDIO.SAMPLE_RATE / sr
                    new_len = int(len(audio) * ratio)
                    audio = np.interp(
                        np.linspace(0, len(audio), new_len),
                        np.arange(len(audio)),
                        audio,
                    ).astype(np.float32)

                audio = preprocess_audio(audio, sr=AUDIO.SAMPLE_RATE)
                mel = computer.compute(audio)
                mel = _pad_or_trim(mel, target_width)
                X_list.append(mel)
                y_list.append(label_idx)

                # Augmentation
                if augment:
                    for _ in range(augment_factor):
                        aug_audio = _augment(audio)
                        aug_mel = computer.compute(aug_audio)
                        aug_mel = _pad_or_trim(aug_mel, target_width)
                        X_list.append(aug_mel)
                        y_list.append(label_idx)
            except Exception as e:
                print(f"    Skipping {fpath.name}: {e}")

    X = np.array(X_list)[..., np.newaxis]  # add channel dim
    y = np.array(y_list)
    print(f"Dataset: {X.shape[0]} samples, shape {X.shape[1:]}")
    return X, y, class_dirs


def _pad_or_trim(mel: np.ndarray, width: int) -> np.ndarray:
    """Ensure the time axis equals *width*."""
    T = mel.shape[1]
    if T >= width:
        return mel[:, :width]
    return np.pad(mel, ((0, 0), (0, width - T)), mode="constant")


def _augment(audio: np.ndarray) -> np.ndarray:
    """Lightweight augmentation: noise injection + amplitude scaling + time shift."""
    aug = audio.copy()

    # Random noise
    noise_level = np.random.uniform(0.001, 0.015)
    aug += np.random.randn(len(aug)).astype(np.float32) * noise_level

    # Amplitude scaling
    aug *= np.random.uniform(0.7, 1.3)

    # Time shift (up to 10 % of length)
    shift = int(np.random.uniform(-0.1, 0.1) * len(aug))
    aug = np.roll(aug, shift)

    # Clip to prevent clipping artefacts
    peak = np.max(np.abs(aug))
    if peak > 1.0:
        aug = aug / peak

    return aug


# ------------------------------------------------------------------ #
# Training
# ------------------------------------------------------------------ #

def train(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    epochs: int = MODEL.EPOCHS,
    batch_size: int = MODEL.BATCH_SIZE,
    lr: float = MODEL.LEARNING_RATE,
    val_split: float = MODEL.VALIDATION_SPLIT,
) -> tuple:
    """Train the CNN and return ``(model, history)``."""
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    num_classes = len(class_names)
    y_cat = to_categorical(y, num_classes=num_classes)

    model = build_cnn_model(
        input_shape=(AUDIO.MEL_HEIGHT, AUDIO.MEL_WIDTH, 1),
        num_classes=num_classes,
        learning_rate=lr,
    )
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    print(f"\nTraining for up to {epochs} epochs (batch_size={batch_size}) ...")
    t0 = time.time()
    history = model.fit(
        X, y_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s")
    return model, history


# ------------------------------------------------------------------ #
# Evaluation
# ------------------------------------------------------------------ #

def evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    val_split: float = MODEL.VALIDATION_SPLIT,
) -> dict:
    """Compute metrics on the held-out validation set and return a dict."""
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        f1_score,
    )

    n = len(X)
    split = int(n * (1 - val_split))
    X_val, y_val = X[split:], y[split:]

    num_classes = len(class_names)
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Use labels param so report works even if some classes are absent in val set
    label_indices = list(range(num_classes))
    report = classification_report(
        y_val, y_pred,
        labels=label_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_val, y_pred, labels=label_indices).tolist()
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    # False positive rate per class
    fpr = {}
    for i, name in enumerate(class_names):
        fp = sum(1 for j, p in enumerate(y_pred) if p == i and y_val[j] != i)
        tn = sum(1 for j, p in enumerate(y_pred) if p != i and y_val[j] != i)
        fpr[name] = round(fp / max(fp + tn, 1), 4)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Classification Report:")
    print(classification_report(
        y_val, y_pred,
        labels=label_indices,
        target_names=class_names,
        zero_division=0,
    ))

    return {
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "false_positive_rate": fpr,
    }


# ------------------------------------------------------------------ #
# Save artefacts
# ------------------------------------------------------------------ #

def save_artefacts(
    model,
    history,
    class_names: list[str],
    metrics: dict,
    out_dir: str = "models",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Keras model
    model_path = os.path.join(out_dir, "cityguard_cnn.keras")
    model.save(model_path)
    print(f"Model saved → {model_path}")

    # Metadata JSON
    meta = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "metrics": metrics,
        "history": {
            "loss": [float(v) for v in history.history.get("loss", [])],
            "accuracy": [float(v) for v in history.history.get("accuracy", [])],
            "val_loss": [float(v) for v in history.history.get("val_loss", [])],
            "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
        },
        "input_shape": list(AUDIO.INPUT_SHAPE),
        "sample_rate": AUDIO.SAMPLE_RATE,
    }
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved → {meta_path}")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Train CityGuard CNN")
    parser.add_argument("--epochs", type=int, default=MODEL.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=MODEL.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=MODEL.LEARNING_RATE)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    # Load
    X, y, class_names = load_dataset(
        base_dir=args.data_dir,
        augment=not args.no_augment,
    )
    if len(X) == 0:
        print("No training data found. Run create_dummy_data.py first.")
        return

    # Train
    model, history = train(
        X, y, class_names,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Evaluate
    metrics = evaluate(model, X, y, class_names)

    # Save
    save_artefacts(model, history, class_names, metrics)
    print("\nDone.")


if __name__ == "__main__":
    main()
