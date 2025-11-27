import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from src.preprocessing import preprocess_single_image
import numpy as np
from PIL import Image
import os
import random

# ---------------------------------------
# RETRAINING PIPELINE
# ---------------------------------------
def retrain_model(existing_model, data_path, save_path):
    datagen = ImageDataGenerator(
        rescale=1/255.,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        data_path,
        target_size=(32, 32),
        batch_size=32,
        class_mode="categorical",
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_path,
        target_size=(32, 32),
        batch_size=32,
        class_mode="categorical",
        subset='validation'
    )

    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    history = existing_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,
        callbacks=[es],
        verbose=1
    )

    existing_model.save(save_path)
    return history, existing_model


# ---------------------------------------
# GRADCAM
# ---------------------------------------
def generate_gradcam(model):
    # Pick a random image
    img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    # Dummy GradCAM (simple heatmap)
    heatmap = np.uint8(np.random.rand(32, 32) * 255)
    heatmap_img = Image.fromarray(heatmap).resize((128, 128))
    return heatmap_img


# ---------------------------------------
# FILTER VISUALIZATION
# ---------------------------------------
def extract_filters(model):
    layer = model.layers[0]
    filters = layer.get_weights()[0]
    filt = filters[:, :, :, 0]

    filt = (filt - filt.min()) / (filt.max() - filt.min())
    filt = np.uint8(filt * 255)

    return Image.fromarray(filt)


# ---------------------------------------
# CONFUSED SAMPLES (RANDOM)
# ---------------------------------------
def get_confused_samples(model):
    img = np.uint8(np.random.rand(32, 32, 3) * 255)
    return Image.fromarray(img)

