"""
src/model.py

Model creation and management utilities for the CIFAR-10 pipeline.

Features:
- create_cifar10_model: CNN architecture matching the notebook
- compile_model: compile helper
- get_training_callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- save_model / load_model: save to .keras format and manage 'latest' alias
- model_summary_str: helper for logs/README

Designed to be robust for local training and for use in the API.
"""

import os
import shutil
import logging
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger("cifar_model")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

MODELS_DIR = "models"
LATEST_MODEL_FILENAME = "cifar10_model_latest.keras"


def create_cifar10_model(input_shape: Tuple[int, int, int] = (32, 32, 3),
                         num_classes: int = 10) -> keras.Model:
    """
    Build the CNN model architecture.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

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

    # Head
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10_cnn")
    logger.debug("Model created.")
    return model


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """
    Compile model with Adam optimizer and categorical crossentropy.
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
    Return EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks.
    """
    os.makedirs(models_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(models_dir, f"cifar10_model_best_{ts}.keras")

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience_es, restore_best_weights=True, verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=reduce_patience, min_lr=min_lr, verbose=1)
    checkpoint = keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1, save_weights_only=False)

    logger.info("Callbacks created. Best-checkpoint will be saved to %s", ckpt_path)
    return [early_stopping, reduce_lr, checkpoint]


def _ensure_models_dir(models_dir: str):
    os.makedirs(models_dir, exist_ok=True)


def save_model(model: keras.Model, models_dir: str = MODELS_DIR, name: Optional[str] = None, make_latest: bool = True) -> str:
    """
    Save model to models_dir. Returns the full path to saved model.
    Uses Keras native format (.keras) or a directory (SavedModel) if provided.
    """
    _ensure_models_dir(models_dir)
    if name:
        model_path = os.path.join(models_dir, name)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(models_dir, f"cifar10_model_{ts}.keras")

    # model.save accepts a filepath ending in .keras, or a folder for SavedModel
    model.save(model_path)
    logger.info("Model saved to %s", model_path)

    if make_latest:
        latest = os.path.join(models_dir, LATEST_MODEL_FILENAME)
        try:
            # copy file or directory
            if os.path.isdir(model_path):
                # SavedModel dir -> copytree (overwrite)
                try:
                    if os.path.exists(latest):
                        if os.path.isdir(latest):
                            shutil.rmtree(latest)
                        else:
                            os.remove(latest)
                    shutil.copytree(model_path, latest)
                except Exception as e:
                    logger.warning("Could not copy SavedModel dir to latest: %s", e)
            else:
                shutil.copy2(model_path, latest)
            logger.info("Updated latest model to %s", latest)
        except Exception as e:
            logger.warning("Failed to update latest model alias: %s", e)

    return model_path


def load_model(model_path: Optional[str] = None, models_dir: str = MODELS_DIR) -> keras.Model:
    """
    Load a model. If model_path is None, tries the latest model alias.
    """
    if model_path is None:
        model_path = os.path.join(models_dir, LATEST_MODEL_FILENAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info("Loading model from %s", model_path)
    m = keras.models.load_model(model_path)
    return m


def model_summary_str(model: Optional[keras.Model] = None) -> str:
    if model is None:
        model = create_cifar10_model()
    lines = []
    model.summary(print_fn=lambda s: lines.append(s))
    return "\n".join(lines)


def _self_test():
    logger.info("Running quick model self-test (small).")
    m = create_cifar10_model()
    compile_model(m)
    path = save_model(m, models_dir=MODELS_DIR, name=f"temp_model_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras", make_latest=False)
    loaded = load_model(path)
    logger.info("Model self-test OK - loaded model type: %s", type(loaded))
    # cleanup
    try:
        if os.path.exists(path) and not os.path.isdir(path):
            os.remove(path)
    except Exception:
        pass
    logger.info("Model self-test completed.")


if __name__ == "__main__":
    _self_test()

