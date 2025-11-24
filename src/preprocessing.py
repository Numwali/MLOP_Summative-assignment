"""
src/preprocessing.py

Data preprocessing utilities for the CIFAR-10 pipeline.

This module provides:
- safe normalization and label handling
- single-image preprocessing for API/UI (accepts bytes or PIL image)
- batch loading utilities from file paths and directory layouts (used for retraining)
- a lightweight augmentation pipeline (tf.keras.layers)
- a small helper to decode prediction arrays into human-readable labels

"""

from typing import Tuple, Optional, List
import os
import io
import logging

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Configure logging for this module
logger = logging.getLogger("preprocessing")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# CIFAR-10 class names (index → label)
CLASS_NAMES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = (32, 32)  # width, height


# ----------------------------
# Basic helper functions
# ----------------------------
def normalize_image_array(x: np.ndarray) -> np.ndarray:
    """
    Normalize an image or batch of images to float32 values in [0, 1].

    Args:
        x: np.ndarray of shape (H, W, C) or (N, H, W, C) with integer pixel values [0..255]

    Returns:
        Normalized float32 array with same shape.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("normalize_image_array expects a numpy array.")
    x = x.astype("float32")
    x /= 255.0
    return x


def to_categorical_labels(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded labels.

    Args:
        y: array-like of integer labels, shape (N,) or (N,1)
        num_classes: total number of classes

    Returns:
        One-hot encoded array shape (N, num_classes)
    """
    y = np.asarray(y).reshape(-1)
    return to_categorical(y, num_classes=num_classes)


# ----------------------------
# Single-image preprocessing (API/UI)
# ----------------------------
def preprocess_single_image(image_bytes: bytes,
                            target_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Preprocess a single uploaded image (bytes) to the model input shape.

    Steps:
    - Load with PIL (handle common formats)
    - Convert to RGB (CIFAR-10 is RGB)
    - Resize to target_size (32x32)
    - Convert to numpy array and normalize
    - Expand dims to (1, H, W, C) for model.predict

    Args:
        image_bytes: raw bytes from uploaded file
        target_size: desired (width, height)

    Returns:
        np.ndarray of shape (1, target_h, target_w, 3), dtype float32
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError as e:
        logger.error("Failed to identify image from bytes: %s", e)
        raise ValueError("Uploaded file is not a valid image.") from e
    except Exception as e:
        logger.exception("Unexpected error loading image bytes.")
        raise

    # Convert to RGB and resize
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, resample=Image.BILINEAR)
    arr = np.asarray(img)
    arr = normalize_image_array(arr)
    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr


# ----------------------------
# Batch loading helpers 
# ----------------------------
def load_images_from_directory(root_dir: str,
                               target_size: Tuple[int, int] = IMG_SIZE,
                               class_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images and labels from a directory with structure:
        root_dir/
            class_name_1/  (or numeric folder '0', '1', ...)
                img1.jpg
                img2.png
            class_name_2/
                ...
    The function tries to map folder names to class indices using `class_names`.
    If the folder name is numeric (e.g., "3"), it is interpreted as the class index.

    Args:
        root_dir: path to the retrain directory
        target_size: (width, height) to resize images to
        class_names: optional list to map class names → indices; defaults to CLASS_NAMES

    Returns:
        Tuple (X, y) where:
            X: np.ndarray of shape (N, H, W, 3), dtype float32 normalized
            y: np.ndarray of shape (N,) of int labels
    """
    if class_names is None:
        class_names = CLASS_NAMES

    X_list = []
    y_list = []

    if not os.path.isdir(root_dir):
        logger.info("Retrain directory %s does not exist or is not a directory.", root_dir)
        return np.empty((0,) + target_size + (3,), dtype="float32"), np.array([], dtype=int)

    # iterate sorted for deterministic order
    for entry in sorted(os.listdir(root_dir)):
        entry_path = os.path.join(root_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        # Resolve class index
        if entry.isdigit():
            cls_idx = int(entry)
        else:
            try:
                cls_idx = class_names.index(entry)
            except ValueError:
                logger.warning("Unknown class folder '%s' — skipping.", entry)
                continue

        # load files
        for fname in sorted(os.listdir(entry_path)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            fpath = os.path.join(entry_path, fname)
            try:
                img = Image.open(fpath)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = img.resize(target_size, resample=Image.BILINEAR)
                arr = np.asarray(img)
                X_list.append(arr)
                y_list.append(cls_idx)
            except Exception as e:
                logger.warning("Skipping file %s due to error: %s", fpath, e)
                continue

    if len(X_list) == 0:
        return np.empty((0,) + target_size + (3,), dtype="float32"), np.array([], dtype=int)

    X = np.stack(X_list, axis=0).astype("float32")
    y = np.array(y_list, dtype=int).reshape(-1)
    X = normalize_image_array(X)
    return X, y


def load_and_preprocess_batch(image_paths: List[str],
                              target_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Load a list of image file paths and return a numpy array ready for prediction:
    shape (N, H, W, 3), dtype float32, normalized.

    Args:
        image_paths: list of file paths
        target_size: desired (width, height)

    Returns:
        np.ndarray of images
    """
    arrs = []
    for p in image_paths:
        try:
            with open(p, "rb") as f:
                b = f.read()
            a = preprocess_single_image(b, target_size=target_size)
            arrs.append(a[0])
        except Exception:
            logger.warning("Failed to load or preprocess %s", p)
            continue
    if len(arrs) == 0:
        return np.empty((0,) + target_size + (3,), dtype="float32")
    return np.stack(arrs, axis=0)


# ----------------------------
# Augmentation helper
# ----------------------------
def get_augmentation_layers() -> tf.keras.Sequential:
    """
    Returns a small tf.keras.Sequential model that performs on-the-fly
    augmentation (use inside model input pipeline for convenience).
    This is deterministic per call unless randomness layers are used.

    Example usage:
        aug = get_augmentation_layers()
        x = aug(inputs)

    Returns:
        tf.keras.Sequential
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.06)
    ], name="data_augmentation_layers")


def get_image_data_generator() -> tf.keras.preprocessing.image.ImageDataGenerator:
    """
    Return an ImageDataGenerator configured for CIFAR-like augmentation.
    Use this for fit() with .flow() if preferred over layers augmentation.

    Note: ImageDataGenerator returns images scaled to [0,1] if rescale=1./255 is set,
    but we prefer explicit normalization functions in code. This function follows
    that pattern for convenience.
    """
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.08,
        rescale=1.0/255.0
    )


# ----------------------------
# Prediction decoding
# ----------------------------
def decode_prediction(pred_array: np.ndarray, class_names: Optional[List[str]] = None) -> Tuple[str, float]:
    """
    Decode a single prediction array (probabilities) to (class_name, confidence).

    Args:
        pred_array: 1-D numpy array of probabilities (len == NUM_CLASSES)
        class_names: optional list of class names

    Returns:
        (class_name, confidence) where confidence is a float in [0,1]
    """
    if class_names is None:
        class_names = CLASS_NAMES
    if pred_array.ndim != 1 or pred_array.shape[0] != NUM_CLASSES:
        logger.warning("decode_prediction received array of shape %s", pred_array.shape)
    idx = int(np.argmax(pred_array))
    conf = float(pred_array[idx])
    name = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
    return name, conf


# ----------------------------
# Quick unit-style checks
# ----------------------------
def _self_test():
    """Quick smoke tests for basic functions (not exhaustive)."""
    logger.info("Running preprocessing self-test...")

    # create fake image
    rand_img = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    norm = normalize_image_array(rand_img)
    assert norm.max() <= 1.0 and norm.min() >= 0.0, "Normalization failed"

    # test single image preprocess (bytes)
    with io.BytesIO() as buf:
        Image.fromarray(rand_img).save(buf, format="PNG")
        b = buf.getvalue()
    pre = preprocess_single_image(b)
    assert pre.shape == (1, 32, 32, 3), f"preprocess_single_image returned shape {pre.shape}"

    # simulate empty retrain dir
    Xr, yr = load_images_from_directory("non_existent_dir_for_test")
    assert Xr.shape[0] == 0 and yr.size == 0

    logger.info("preprocessing self-test passed.")


if __name__ == "__main__":
    _self_test()

