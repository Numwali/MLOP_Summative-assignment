"""
src/model.py

Clean and production-ready model utilities for CIFAR-10.
"""

import os
import shutil
import logging
from datetime import datetime
from typing import Optional, Tuple, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Logging setup
logger = logging.getLogger("model")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default directories
MODELS_DIR = "models"
LATEST_MODEL_NAME = "cifar10_model_latest.keras"


# ----------------------------------------------------------
# MODEL ARCHITECTURE
# ----------------------------------------------------------
def create_cifar10_model(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10
) -> keras.Model:
    """
    Builds the CNN architecture used for CIFAR-10.
    """
    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.4)(x)

    # Dense head
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cifar10_cnn")
    logger.info("CIFAR-10 model built successfully.")
    return model


# ----------------------------------------------------------
# COMPILE MODEL
# ----------------------------------------------------------
def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """
    Compiles model with Adam + categorical crossentropy.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    logger.info("Model compiled. LR=%.6f", learning_rate)
    return model


# ----------------------------------------------------------
# CALLBACKS
# ----------------------------------------------------------
def get_callbacks(models_dir: str = MODELS_DIR) -> List[keras.callbacks.Callback]:
    """
    Returns EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
    """
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_path = os.path.join(models_dir, f"checkpoint_{timestamp}.keras")

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
    ]


# ----------------------------------------------------------
# SAVE & LOAD MODEL
# ----------------------------------------------------------
def save_model(model: keras.Model,
               name: Optional[str] = None,
               models_dir: str = MODELS_DIR,
               make_latest: bool = True) -> str:
    """
    Saves the model in `.keras` format.
    Also updates the `latest` model alias.
    """
    os.makedirs(models_dir, exist_ok=True)

    if name:
        file_path = os.path.join(models_dir, name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(models_dir, f"cifar10_model_{timestamp}.keras")

    model.save(file_path)
    logger.info("Model saved at %s", file_path)

    if make_latest:
        latest_path = os.path.join(models_dir, LATEST_MODEL_NAME)
        shutil.copy2(file_path, latest_path)
        logger.info("Updated latest model -> %s", latest_path)

    return file_path


def load_model(path: Optional[str] = None,
               models_dir: str = MODELS_DIR) -> keras.Model:
    """
    Loads the model from disk.
    If no path provided → loads the 'latest' model.
    """
    if path is None:
        path = os.path.join(models_dir, LATEST_MODEL_NAME)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")

    logger.info("Loading model from %s", path)
    model = keras.models.load_model(path)
    logger.info("Model loaded successfully.")
    return model


# ----------------------------------------------------------
# SUMMARY STRING
# ----------------------------------------------------------
def model_summary_str(model: Optional[keras.Model] = None) -> str:
    """
    Returns the model summary as a clean string.
    """
    if model is None:
        model = create_cifar10_model()

    lines = []
    model.summary(print_fn=lambda s: lines.append(s))
    return "\n".join(lines)


# ----------------------------------------------------------
# SELF TEST
# ----------------------------------------------------------
if __name__ == "__main__":
    logger.info("Running model.py self-test…")
    try:
        m = create_cifar10_model()
        m = compile_model(m)
        p = save_model(m, make_latest=False)
        _ = load_model(p)
        os.remove(p)
        logger.info("Self-test passed.")
    except Exception as e:
        logger.error("Self-test failed: %s", e)

