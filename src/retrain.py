# src/retrain.py
import os
import shutil
from datetime import datetime
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from src.model import create_cnn_model, compile_model
from tensorflow.keras.utils import to_categorical
from src.preprocessing import load_cifar10
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

def retrain_from_directory(retrain_dir='data/train', epochs=3, batch_size=32, backup=True, latest_model_path='models/cifar10_model_latest.keras'):
    if not os.path.isdir(retrain_dir):
        return {"error": "retrain_dir does not exist"}

    # Count images
    total_images = 0
    for root, dirs, files in os.walk(retrain_dir):
        total_images += sum(1 for f in files if f.lower().endswith(('.png','.jpg','.jpeg')))
    if total_images == 0:
        return {"error": "No images found in retrain_dir"}

    # Backup existing model
    if backup and os.path.exists(latest_model_path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"models/cifar10_model_backup_{ts}.keras"
        shutil.copy2(latest_model_path, backup_path)

    # Load or create model
    if os.path.exists(latest_model_path):
        model = load_model(latest_model_path)
    else:
        model = create_cnn_model()
        model = compile_model(model, lr=1e-4)

    # Data generators
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.1
    )

    train_gen = datagen.flow_from_directory(
        retrain_dir,
        target_size=(32,32),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        retrain_dir,
        target_size=(32,32),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # compile with low lr for fine-tune
    try:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    except Exception:
        model = compile_model(model, lr=1e-4)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
    ]

    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks, verbose=1)

    # save model
    os.makedirs(os.path.dirname(latest_model_path), exist_ok=True)
    model.save(latest_model_path)

    retrain_log = {
        "timestamp": datetime.now().isoformat(),
        "total_images": total_images,
        "history": {k: [float(x) for x in v] for k,v in history.history.items()},
        "save_path": latest_model_path
    }
    # write retrain log
    os.makedirs('logs', exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"logs/retrain_log_{ts}.json", 'w') as f:
        json.dump(retrain_log, f, indent=2)

    return retrain_log


def evaluate_model_on_testset(model_path='models/cifar10_model_latest.keras'):
    """Load CIFAR-10 test set and return accuracy, precision, recall, f1 (weighted)."""
    from tensorflow.keras.models import load_model as tf_load_model
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model path not found for evaluation")

    model = tf_load_model(model_path)
    (X_train, y_train), (X_test, y_test) = load_cifar10()
    X_test = X_test.astype('float32') / 255.0

    y_probs = model.predict(X_test, batch_size=256)
    y_pred = y_probs.argmax(axis=1)
    y_true = y_test.flatten()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    rec = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    return metrics
