"""
src/preprocessing.py

Image preprocessing utilities for the CIFAR-10 pipeline.

Provides:
- normalize_image_array: normalize images to [0,1]
- preprocess_single_image: accept image bytes and return (1,32,32,3) float32 array
- load_images_from_directory: load folder-structured images for retraining
- preprocess_data: bulk preprocess for training (X normalized, y one-hot)
- CLASS_NAMES: CIFAR-10 labels and decode_prediction helper

This module is intentionally small and dependency-light so it can be used by
both the notebook and the API without modification.
"""

import os
import io
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def normalize_image_array(img_array: np.ndarray) -> np.ndarray:
    """
    Normalize a uint8 image array to float32 in [0, 1].

    Args:
        img_array: np.ndarray, shape (..., H, W, C) or (H, W, C)

    Returns:
        float32 array with same shape, values in [0,1]
    """
    if img_array.dtype != np.float32:
        img_array = img_array.astype("float32")
    return img_array / 255.0


def preprocess_single_image(img_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to a preprocessed (1, 32, 32, 3) array ready for model input.

    Args:
        img_bytes: bytes (uploaded file)

    Returns:
        x: np.ndarray shape (1,32,32,3), dtype float32
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((32, 32))
    arr = np.array(img)
    arr = normalize_image_array(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def load_images_from_directory(directory: str):
    """
    Load images from a folder where each class has a sub-folder, e.g. 'cat/', 'dog/'.

    Args:
        directory: path to retrain directory. Expected structure:
           directory/
              class_a/
                 img1.jpg
                 img2.png
              class_b/
                 ...
    Returns:
        X: np.ndarray of shape (N, 32, 32, 3) dtype uint8
        y: np.ndarray of shape (N,) of integer labels (indices in CLASS_NAMES)
    """
    images = []
    labels = []

    if not os.path.exists(directory):
        logger.debug("load_images_from_directory: directory does not exist: %s", directory)
        return np.array(images), np.array(labels)

    for class_name in sorted(os.listdir(directory)):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_name not in CLASS_NAMES:
            logger.warning("Unknown class folder '%s' - skipping (expected one of %s)", class_name, CLASS_NAMES)
            continue
        class_idx = CLASS_NAMES.index(class_name)
        for fname in sorted(os.listdir(class_dir)):
            fpath = os.path.join(class_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB").resize((32, 32))
                images.append(np.array(img))
                labels.append(class_idx)
            except Exception as e:
                logger.warning("Failed to load %s: %s", fpath, e)

    if len(images) == 0:
        return np.array(images), np.array(labels)

    X = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int32)
    return X, y


def preprocess_data(X, y):
    """
    Preprocess training data: normalize X and one-hot encode y (if needed).

    Args:
        X: np.ndarray (N, 32, 32, 3) or (32,32,3)
        y: np.ndarray shape (N,) or (N,1)

    Returns:
        X_proc, y_proc  -- X float32 normalized, y one-hot shape (N, num_classes)
    """
    X_proc = normalize_image_array(X)
    # handle shape quirks
    y_arr = np.array(y)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr.flatten()
    # If labels are strings (rare), attempt mapping to indices
    if y_arr.dtype.type is np.str_:
        label_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
        y_mapped = np.array([label_to_idx.get(v, -1) for v in y_arr], dtype=np.int32)
        y_arr = y_mapped

    y_proc = to_categorical(y_arr, num_classes=len(CLASS_NAMES))
    return X_proc, y_proc


def decode_prediction(pred_array):
    """
    Convert a single prediction probability vector to (class_name, confidence).
    """
    idx = int(np.argmax(pred_array))
    conf = float(pred_array[idx])
    return CLASS_NAMES[idx], conf


# minimal self-test when invoked directly
def _self_test():
    rand_img = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    norm = normalize_image_array(rand_img)
    assert 0.0 <= norm.min() <= norm.max() <= 1.0
    b = None
    with io.BytesIO() as buf:
        Image.fromarray(rand_img).save(buf, format="PNG")
        b = buf.getvalue()
    p = preprocess_single_image(b)
    assert p.shape == (1, 32, 32, 3)
    Xr, yr = load_images_from_directory("non_existent_dir_for_test")
    assert Xr.size == 0 and yr.size == 0
    print("preprocessing self-test passed.")


if __name__ == "__main__":
    _self_test()

