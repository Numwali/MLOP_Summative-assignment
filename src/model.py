"""
src/model.py

CNN model utilities for CIFAR-10.
Handles:

* Model architecture
* Compilation
* Callbacks
* Safe saving & loading in native `.keras` format
"""

import os
import shutil
import logging
from datetime import datetime
from typing import Optional, Tuple, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Logger setup
logger = logging.getLogger("model")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

MODELS_DIR = "models"
LATEST_MODEL = "cifar10_model_latest.keras"

# -----------------------------------------------------
# Model Architecture
# -----------------------------------------------------
def create_cifar10_model(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10
) -> keras.Model:

    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

    # Dense head
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="cifar10_cnn")

# -----------------------------------------------------
# Compile Helper
# -----------------------------------------------------
def compile_model(model: keras.Model, learning_rate: float = 1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info(f"Model compiled. LR={learning_rate}")
    return model

# -----------------------------------------------------
# Save & Load Utilities
# -----------------------------------------------------
def save_model(model: keras.Model, filename: str = LATEST_MODEL) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)

    full_path = os.path.join(MODELS_DIR, filename)
    model.save(full_path)

    logger.info(f"Model saved: {full_path}")
    return full_path


def load_model(path: str = None) -> keras.Model:
    if path is None:
        path = os.path.join(MODELS_DIR, LATEST_MODEL)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")

    logger.info(f"Loading model: {path}")
    return keras.models.load_model(path)

