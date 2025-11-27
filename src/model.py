"""
model.py
Contains model creation, compile, train_and_save utilities.
"""
import os
from datetime import datetime
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def create_cnn_model(input_shape=(32,32,3), num_classes=10):
    """Create and return a compiled Keras model (not compiled here).
    Caller should compile with desired optimizer and loss.
    """
    inputs = keras.Input(shape=input_shape)
    # augmentation should be inserted by the training code if desired
    x = layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='cifar10_cnn')
    return model


def compile_model(model: keras.Model, lr: float = 1e-3):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_and_save(model: keras.Model, X_train, y_train, X_val, y_val, save_dir='models',
                   batch_size=128, epochs=20, callbacks=None, save_latest=True) -> Tuple[keras.Model, dict]:
    """Train model and save best + latest weights.
    Returns (model, history.history dict)
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_path = os.path.join(save_dir, f'cifar10_model_best_{timestamp}.keras')
    final_path = os.path.join(save_dir, f'cifar10_model_{timestamp}.keras')
    latest_path = os.path.join(save_dir, 'cifar10_model_latest.keras')

    # Ensure ModelCheckpoint is present
    cb = callbacks or []
    has_checkpoint = any(isinstance(c, keras.callbacks.ModelCheckpoint) for c in cb)
    if not has_checkpoint:
        cb.append(keras.callbacks.ModelCheckpoint(best_path, monitor='val_loss', save_best_only=True))

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=cb)

    # save final and latest
    model.save(final_path)
    if save_latest:
        model.save(latest_path)
    return model, history.history


def load_model_from_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return keras.models.load_model(path)

