from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from config import AUDIO, CLASS_NAMES, MODEL
from models.cnn_model import build_cnn_model, build_crnn_model
from preprocessing.audio import preprocess_audio
from preprocessing.mel_spectrogram import MelSpectrogramComputer


def _pad_or_trim_mel(mel: np.ndarray, target_width: int = AUDIO.MEL_WIDTH) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim != 2:
        raise ValueError(f"Expected a 2D mel spectrogram, got shape {mel.shape}.")
    if mel.shape[1] >= target_width:
        mel = mel[:, :target_width]
    else:
        pad = target_width - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad)), mode="constant")
    return mel


def load_dataset(data_dir: str = ".") -> tuple[np.ndarray, np.ndarray, list[str]]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    X: list[np.ndarray] = []
    y: list[int] = []
    computer = MelSpectrogramComputer(
        sr=AUDIO.SAMPLE_RATE,
        n_fft=AUDIO.N_FFT,
        hop_length=AUDIO.HOP_LENGTH,
        n_mels=AUDIO.N_MELS,
        fmin=AUDIO.FMIN,
        fmax=AUDIO.FMAX,
    )

    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = data_path / class_name
        if not class_dir.exists():
            continue
        for wav_path in sorted(class_dir.glob("*.wav")):
            audio, sr = sf.read(wav_path, dtype="float32")
            audio = preprocess_audio(audio, sr=sr, target_sr=AUDIO.SAMPLE_RATE, window_length=int(AUDIO.SAMPLE_RATE * AUDIO.WINDOW_DURATION))
            mel = computer.compute(audio)
            mel = _pad_or_trim_mel(mel, target_width=AUDIO.MEL_WIDTH)
            X.append(mel[..., np.newaxis])
            y.append(class_id)

    if not X:
        raise FileNotFoundError(f"No audio files were found under {data_path}")

    return np.stack(X, axis=0).astype(np.float32), np.asarray(y, dtype=np.int32), CLASS_NAMES


def _augment_training_mels(X: np.ndarray, y: np.ndarray, copies: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Augment training examples only; validation and test arrays are untouched."""
    augmented = [X]
    labels = [y]
    for copy_index in range(copies):
        rng = np.random.default_rng(MODEL.RANDOM_SEED + copy_index)
        noisy = X + rng.normal(0.0, 0.015, size=X.shape).astype(np.float32)
        shift = int(rng.integers(-4, 5))
        shifted = np.roll(noisy, shift=shift, axis=2)
        augmented.append(shifted.astype(np.float32))
        labels.append(y)
    return np.concatenate(augmented), np.concatenate(labels)


def train_model(
    model_name: str,
    epochs: int = MODEL.EPOCHS,
    batch_size: int = MODEL.BATCH_SIZE,
    data_dir: str = ".",
) -> tuple[tf.keras.Model, dict]:
    tf.keras.utils.set_random_seed(MODEL.RANDOM_SEED)
    X, y, names = load_dataset(data_dir)
    class_counts = np.bincount(y, minlength=len(names))
    if np.any(class_counts < 2):
        raise ValueError(
            "Each class needs at least two source recordings for a stratified validation split. "
            f"Counts: {dict(zip(names, class_counts.tolist()))}"
        )

    X_train, X_val, y_train_ids, y_val_ids = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=MODEL.RANDOM_SEED,
        stratify=y,
    )
    X_train, y_train_ids = _augment_training_mels(
        X_train,
        y_train_ids,
        copies=MODEL.AUGMENTATION_FACTOR,
    )
    y_train = tf.keras.utils.to_categorical(y_train_ids, num_classes=len(names))
    y_val = tf.keras.utils.to_categorical(y_val_ids, num_classes=len(names))

    if model_name == "cnn":
        model = build_cnn_model(input_shape=AUDIO.INPUT_SHAPE, num_classes=len(names), learning_rate=MODEL.LEARNING_RATE)
        save_name = "cityguard_cnn.keras"
    elif model_name == "crnn":
        model = build_crnn_model(input_shape=AUDIO.INPUT_SHAPE, num_classes=len(names), learning_rate=MODEL.LEARNING_RATE)
        save_name = "cityguard_crnn.keras"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path("models") / f"{model_name}.best.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    model_path = Path("models") / save_name
    model_path.parent.mkdir(exist_ok=True)
    model.save(model_path)

    return model, {
        "history": history.history,
        "model_name": save_name,
        "class_names": names,
        "sample_rate": AUDIO.SAMPLE_RATE,
        "window_duration": AUDIO.WINDOW_DURATION,
        "n_mels": AUDIO.N_MELS,
        "n_fft": AUDIO.N_FFT,
        "hop_length": AUDIO.HOP_LENGTH,
        "train_count": int(len(X_train)),
        "validation_count": int(len(X_val)),
        "class_counts": dict(zip(names, class_counts.tolist())),
        "validation_labels": y_val_ids.tolist(),
    }


def evaluate_model(model: tf.keras.Model, data_dir: str = ".") -> dict:
    X, y, names = load_dataset(data_dir)
    _, X, _, y = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=MODEL.RANDOM_SEED,
        stratify=y,
    )
    probs = model.predict(X, verbose=0)
    pred = np.argmax(probs, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=list(range(len(names))), average="macro", zero_division=0)
    acc = accuracy_score(y, pred)
    weighted_f1 = f1_score(y, pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(len(names)))).tolist()
    per_class = {
        names[i]: {
            "precision": float(precision_recall_fscore_support(y, pred, labels=[i], zero_division=0)[0][0]),
            "recall": float(precision_recall_fscore_support(y, pred, labels=[i], zero_division=0)[1][0]),
            "f1": float(precision_recall_fscore_support(y, pred, labels=[i], zero_division=0)[2][0]),
        }
        for i in range(len(names))
    }

    return {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
        "classes": names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the CityGuard models.")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "crnn"])
    parser.add_argument("--epochs", type=int, default=MODEL.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=MODEL.BATCH_SIZE)
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    model, info = train_model(args.model, epochs=args.epochs, batch_size=args.batch_size, data_dir=args.data_dir)
    metadata = {"model": args.model, **info}
    Path("results").mkdir(exist_ok=True)
    Path("results/metadata.json").write_text(json.dumps(metadata, indent=2))

    metrics = evaluate_model(model, data_dir=args.data_dir)
    Path(f"results/{args.model}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    print(f"Model saved to models/{args.model == 'cnn' and 'cityguard_cnn.keras' or 'cityguard_crnn.keras'}")


if __name__ == "__main__":
    main()
