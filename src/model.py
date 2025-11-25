"""
src/model.py

Model creation and management utilities for the CIFAR-10 pipeline.

- Build CNN matching the notebook architecture
- Compile helper
- Training callbacks
- Save / load helpers using native Keras format (.keras)
- Small self-test utilities
"""

import os
import shutil
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger("src.model")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# Defaults
MODELS_DIR = "models"
LATEST_MODEL_FILENAME = "cifar10_model_latest.keras"


def create_cifar10_model(input_shape: Tuple[int, int, int] = (32, 32, 3),
                         num_classes: int = 10) -> keras.Model:
    """
    Build a CNN model for CIFAR-10.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    # Head
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10_cnn")
    return model


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """
    Compile the model with Adam and categorical crossentropy.
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    logger.info("Model compiled. LR=%.6f", learning_rate)
    return model


def get_training_callbacks(models_dir: str = MODELS_DIR,
                           patience_es: int = 3,
                           reduce_patience: int = 2,
                           min_lr: float = 1e-7) -> List[keras.callbacks.Callback]:
    """
    Return common callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
    The checkpoint is saved in `models_dir` with a timestamp.
    """
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(models_dir, f"checkpoint_val_loss_{timestamp}.keras")

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=patience_es,
                                                   restore_best_weights=True,
                                                   verbose=1)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                  factor=0.5,
                                                  patience=reduce_patience,
                                                  min_lr=min_lr,
                                                  verbose=1)

    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 monitor="val_loss",
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 verbose=1)

    logger.info("Callbacks created. Checkpoint path: %s", checkpoint_path)
    return [early_stopping, reduce_lr, checkpoint]


def _ensure_models_dir(models_dir: str):
    os.makedirs(models_dir, exist_ok=True)


def save_model(model: keras.Model,
               models_dir: str = MODELS_DIR,
               name: Optional[str] = None,
               make_latest: bool = True) -> str:
    """
    Save the model in native Keras format (.keras). Returns full path to saved model.
    If make_latest=True, copy the file to models_dir/cifar10_model_latest.keras
    """
    _ensure_models_dir(models_dir)

    if name:
        file_path = os.path.join(models_dir, name)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(models_dir, f"cifar10_model_{ts}.keras")

    # model.save accepts both file (.keras) and directory (SavedModel)
    model.save(file_path)  # native Keras format
    logger.info("Model saved: %s", file_path)

    if make_latest:
        latest = os.path.join(models_dir, LATEST_MODEL_FILENAME)
        try:
            shutil.copy2(file_path, latest)
            logger.info("Updated latest model: %s", latest)
        except Exception as e:
            logger.warning("Could not update latest alias: %s", e)

    return file_path


def load_model(model_path: Optional[str] = None, models_dir: str = MODELS_DIR) -> keras.Model:
    """
    Load a model. If model_path is None, load models/LATEST_MODEL_FILENAME.
    Raises FileNotFoundError if missing.
    """
    if model_path is None:
        model_path = os.path.join(models_dir, LATEST_MODEL_FILENAME)

    if not os.path.exists(model_path):
        logger.error("Model not found at %s", model_path)
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info("Loading model from %s", model_path)
    m = keras.models.load_model(model_path)
    return m


def model_summary_str(model: Optional[keras.Model] = None) -> str:
    if model is None:
        model = create_cifar10_model()
    lines: List[str] = []
    model.summary(print_fn=lambda s: lines.append(s))
    return "\n".join(lines)


def _self_test():
    logger.info("Running model self-test (small smoke tests)...")
    m = create_cifar10_model()
    m = compile_model(m)
    p = save_model(m, models_dir=MODELS_DIR, make_latest=False)
    loaded = load_model(p)
    os.remove(p)
    logger.info("Model self-test passed.")


if __name__ == "__main__":
    _self_test()

