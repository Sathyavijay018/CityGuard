from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import AUDIO, CLASS_NAMES


def build_cnn_model(
    input_shape: tuple[int, int, int] = AUDIO.INPUT_SHAPE,
    num_classes: int = len(CLASS_NAMES),
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Lightweight CNN baseline for 4-class acoustic classification."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_logits")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cityguard_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_crnn_model(
    input_shape: tuple[int, int, int] = AUDIO.INPUT_SHAPE,
    num_classes: int = len(CLASS_NAMES),
    learning_rate: float = 1e-3,
) -> keras.Model:
    """CNN-GRU model that preserves time structure for temporal reasoning."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    time_steps = x.shape[2]
    feature_dim = x.shape[1] * x.shape[3]
    x = layers.Reshape((-1, feature_dim))(x)
    x = layers.GRU(64, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cityguard_crnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _prepare_mel_input(mel_spec: np.ndarray, target_shape: tuple[int, int, int] = AUDIO.INPUT_SHAPE) -> np.ndarray:
    arr = np.asarray(mel_spec, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 2:
        arr = arr[: target_shape[0], : target_shape[1]]
        if arr.shape[1] < target_shape[1]:
            arr = np.pad(arr, ((0, target_shape[0] - arr.shape[0]), (0, target_shape[1] - arr.shape[1])), mode="constant")
        if arr.shape[0] < target_shape[0]:
            arr = np.pad(arr, ((0, target_shape[0] - arr.shape[0]), (0, 0)), mode="constant")
        arr = arr[..., np.newaxis]
    elif arr.shape[-1] != 1:
        arr = np.moveaxis(arr, -1, 0)[..., np.newaxis]
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)


def predict_single(model: keras.Model, mel_spec: np.ndarray, class_names: list[str] | None = None) -> tuple[str, float, dict]:
    """Return predicted class, confidence, and per-class probabilities for a single mel spectrogram."""
    if mel_spec is None:
        raise ValueError("Mel spectrogram is None.")

    arr = _prepare_mel_input(mel_spec)
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    names = class_names or CLASS_NAMES
    class_name = names[pred_idx]
    confidence = float(probs[pred_idx])
    return class_name, confidence, {names[i]: float(probs[i]) for i in range(len(names))}


def load_model(model_path: str | Path, class_names: list[str] | None = None) -> tuple[keras.Model, list[str]]:
    """Load a saved Keras model, or return a freshly built CNN if absent."""
    path = Path(model_path)
    model_names = class_names or CLASS_NAMES
    if path.exists():
        model = keras.models.load_model(str(path))
        return model, model_names

    model = build_cnn_model(input_shape=AUDIO.INPUT_SHAPE, num_classes=len(model_names))
    return model, model_names
