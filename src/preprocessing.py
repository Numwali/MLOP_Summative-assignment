---------
"""
preprocessing.py
Helpers for loading images, preprocessing arrays, and augmentation.
"""
import os
from typing import Tuple
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

IMG_SIZE = (32, 32)
IMG_SHAPE = (32, 32, 3)


def load_cifar10() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load CIFAR-10 via Keras and return ((x_train,y_train),(x_test,y_test)).
    Labels returned as 1D arrays.
    """
    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return (x_train, y_train), (x_test, y_test)


def preprocess_arrays(X: np.ndarray, y: np.ndarray, num_classes: int):
    """Normalize X to [0,1] and one-hot encode labels y.
    Returns (Xp, yp)
    """
    from tensorflow.keras.utils import to_categorical
    Xp = X.astype("float32") / 255.0
    yp = to_categorical(y, num_classes)
    return Xp, yp


def get_augmentation_layer():
    """Return a Keras Sequential of augmentation layers.
    This is lightweight and suitable for CIFAR-10-sized images.
    """
    from tensorflow.keras import layers, Sequential
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.06),
    ], name="data_augmentation")
    return data_augmentation


def preprocess_single_image_from_file(path: str, target_size=IMG_SIZE) -> np.ndarray:
    """Load an image file and return a HxWx3 numpy array (uint8) resized to target_size."""
    img = load_img(path, target_sitrain_dir,
        'new_model': newpath,
        'history': {k: [float(x) for x in v] for k, v in history.history.items()}
    }
    import json
    with open(os.path.join('logs', f'retrain_log_{ts}.json'), 'w') as f:
        json.dump(retrain_log, f, indent=2)
    print('[retrain] retrain log saved to logs/')
    return retrain_log
