"""
src/model.py

Model creation and management utilities for the CIFAR-10 pipeline.

This module provides:
- A configurable CNN architecture (matches the notebook)
- Compile helper with optimizer / loss / metrics
- Training callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- Robust save/load helpers with safe directories and optional versioning
- A helper to return the model summary as a string (useful for logs/README)

"""

import os
import shutil
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configure module logger
logger = logging.getLogger("model")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# Default paths
MODELS_DIR = "models"
LATEST_MODEL_FILENAME = "cifar10_model_latest.h5"
BEST_MODEL_FILENAME = "cifar10_model_best.h5"


# ----------------------------
# Model architecture
# ----------------------------
def create_cifar10_model(input_shape: Tuple[int, int, int] = (32, 32, 3),
                         num_classes: int = 10) -> keras.Model:
    """
    Build a CNN model for CIFAR-10.

    Architecture:
      - 3 convolutional blocks (Conv2D -> BatchNorm -> Conv2D -> BatchNorm -> MaxPool -> Dropout)
      - Dense head with BatchNorm + Dropout
      - Softmax output for `num_classes`

    Args:
        input_shape: input image shape
        num_classes: number of classes

    Returns:
        Compiled Keras model (not compiled here — see compile_model)
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Block 1
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.4)(x)

    # Dense head
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10_cnn")
    return model


# ----------------------------
# Compile helper
# ----------------------------
def compile_model(model: keras.Model,
                  learning_rate: float = 1e-3) -> keras.Model:
    """
    Compile model with Adam optimizer and categorical crossentropy loss.
    Adds accuracy metric by default.

    Args:
        model: Keras model
        learning_rate: learning rate for Adam

    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    logger.info("Compiled model with Adam lr=%.6f", learning_rate)
    return model


# ----------------------------
# Callbacks
# ----------------------------
def get_training_callbacks(models_dir: str = MODELS_DIR,
                           patience_es: int = 3,
                           reduce_patience: int = 2,
                           min_lr: float = 1e-7) -> List[keras.callbacks.Callback]:
    """
    Create a set of callbacks for training:
      - EarlyStopping (monitors val_loss)
      - ReduceLROnPlateau (monitors val_loss)
      - ModelCheckpoint (saves best model by val_loss)

    Args:
        models_dir: directory where to save checkpoint
        patience_es: patience for EarlyStopping
        reduce_patience: patience for ReduceLROnPlateau
        min_lr: minimum learning rate

    Returns:
        list of keras callbacks
    """
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(models_dir, f"checkpoint_val_loss_{timestamp}.h5")

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience_es,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=reduce_patience,
        min_lr=min_lr,
        verbose=1
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    logger.info("Callbacks created. Checkpoint will be saved to %s", checkpoint_path)
    return [early_stopping, reduce_lr, checkpoint]


# ----------------------------
# Save and load helpers
# ----------------------------
def _ensure_models_dir(models_dir: str):
    os.makedirs(models_dir, exist_ok=True)


def save_model(model: keras.Model,
               models_dir: str = MODELS_DIR,
               name: Optional[str] = None,
               make_latest: bool = True) -> str:
    """
    Save model to disk. If name is None, a timestamped name is created.
    Optionally update the 'latest' alias.

    Args:
        model: keras Model
        models_dir: directory to store models
        name: optional filename (e.g. 'my_model.h5')
        make_latest: whether to copy the saved file to the 'latest' filename

    Returns:
        The path to the saved model file.
    """
    _ensure_models_dir(models_dir)
    if name:
        model_path = os.path.join(models_dir, name)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(models_dir, f"cifar10_model_{ts}.h5")

    model.save(model_path)
    logger.info("Model saved to %s", model_path)

    if make_latest:
        latest = os.path.join(models_dir, LATEST_MODEL_FILENAME)
        try:
            shutil.copy2(model_path, latest)
            logger.info("Updated latest model to %s", latest)
        except Exception as e:
            logger.warning("Could not update latest model alias: %s", e)

    return model_path


def load_model(model_path: Optional[str] = None,
               models_dir: str = MODELS_DIR) -> keras.Model:
    """
    Load a Keras model from disk.

    If model_path is None, it attempts to load the 'latest' model.

    Args:
        model_path: path to .h5 or SavedModel directory, or None
        models_dir: directory where latest model may be stored

    Returns:
        Loaded keras.Model
    """
    if model_path is None:
        model_path = os.path.join(models_dir, LATEST_MODEL_FILENAME)

    if not os.path.exists(model_path):
        logger.error("Model file not found at %s", model_path)
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info("Loading model from %s", model_path)
    m = keras.models.load_model(model_path)
    return m


# ----------------------------
# Utility: model summary as string
# ----------------------------
def model_summary_str(model: Optional[keras.Model] = None) -> str:
    """
    Return the model summary (text) for logging or README.

    Args:
        model: If None, create a fresh uncompiled model with default shape

    Returns:
        Summary string
    """
    if model is None:
        model = create_cifar10_model()
    summary_lines: List[str] = []
    model.summary(print_fn=lambda s: summary_lines.append(s))
    return "\n".join(summary_lines)


# ----------------------------
# Quick smoke test 
# ----------------------------
def _self_test():
    """
    Run lightweight tests: build, compile, save, load, and cleanup.
    Use this to sanity-check the environment.
    """
    logger.info("Running model self-test...")
    try:
        m = create_cifar10_model()
        m = compile_model(m, learning_rate=1e-3)
        logger.info("Model created and compiled.")
        # small summary
        s = model_summary_str(m)
        logger.info("Model summary length: %d chars", len(s))

        # Save and load test (temporary)
        p = save_model(m, models_dir=MODELS_DIR, make_latest=False)
        logger.info("Saved temporary model to %s", p)
        loaded = load_model(p)
        logger.info("Loaded model, OK.")
        # cleanup the temp model file
        try:
            os.remove(p)
            logger.info("Removed temporary model file %s", p)
        except Exception:
            pass

        logger.info("Model self-test passed.")
    except Exception as e:
        logger.exception("Model self-test failed: %s", e)
        raise


if __name__ == "__main__":
    _self_test()

