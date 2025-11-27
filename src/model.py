# src/model.py

import os
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def create_cnn_model(input_shape=(32,32,3), num_classes=10):
    """Create CIFAR-10 CNN architecture."""
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_and_save(model, X_train, y_train, X_val, y_val, save_dir="models",
                   epochs=20, batch_size=128, callbacks=None):

    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_path = os.path.join(save_dir, f"cifar10_best_{ts}.keras")
    latest_path = os.path.join(save_dir, "cifar10_model_latest.keras")

    cb = callbacks or []
    cb.append(
        keras.callbacks.ModelCheckpoint(best_path, save_best_only=True, monitor="val_loss")
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=cb
    )

    model.save(latest_path)
    return model, history.history
