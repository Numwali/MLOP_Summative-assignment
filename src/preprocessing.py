import numpy as np
from tensorflow.keras.utils import to_categorical


NUM_CLASSES = 10
IMG_SHAPE = (32,32,3)


def preprocess_arrays(X, y):
Xp = X.astype('float32') / 255.0
yp = to_categorical(y, NUM_CLASSES)
return Xp, yp


# Data augmentation layers
from tensorflow.keras import layers, Sequential
data_augmentation = Sequential([
layers.RandomFlip("horizontal"),
layers.RandomRotation(0.06),
layers.RandomZoom(0.06),
], name="data_augmentation")
