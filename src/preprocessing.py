# src/preprocessing.py

import numpy as np
from typing import Tuple

IMG_SIZE = (32, 32)
IMG_SHAPE = (32, 32, 3)


def load_cifar10():
    """Load CIFAR-10 dataset."""
    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (x_train, y_train.flatten()), (x_test, y_test.flatten())


def preprocess_arrays(X, y, num_classes):
    """Normalize images + one hot encode labels."""
    from tensorflow.keras.utils import to_categorical
    Xp = X.astype("float32") / 255.0
    yp = to_categorical(y, num_classes)
    return Xp, yp


def get_augmentation_layer():
    """Return lightweight augmentation block."""
    from tensorflow.keras import layers, Sequential
    return Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ])
