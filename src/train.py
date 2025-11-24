"""
Training module for CIFAR-10 model
Handles initial training and retraining with full logging and metrics.
"""

import numpy as np
import json
import os
from datetime import datetime
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.model import create_cifar10_model, compile_model, save_model, get_training_callbacks
from src.preprocessing import preprocess_data, create_data_augmentation

def train_model(epochs=50, batch_size=128, validation_split=0.2, use_augmentation=True):
"""
Train CIFAR-10 model from scratch.

```
Args:
    epochs: Number of training epochs
    batch_size: Batch size for training
    validation_split: Fraction of data for validation
    use_augmentation: Whether to use data augmentation

Returns:
    model: Trained Keras model
    history: Training history object
"""
print("Loading CIFAR-10 dataset...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Preprocessing data...")
X_train_processed, y_train_processed = preprocess_data(X_train, y_train)
X_test_processed, y_test_processed = preprocess_data(X_test, y_test)

print("Creating and compiling model...")
model = create_cifar10_model()
model = compile_model(model)

callbacks = get_training_callbacks()

if use_augmentation:
    print("Using data augmentation...")
    datagen = create_data_augmentation()
    datagen.fit(X_train_processed)
    
    val_size = int(len(X_train_processed) * validation_split)
    X_val = X_train_processed[-val_size:]
    y_val = y_train_processed[-val_size:]
    X_train_fit = X_train_processed[:-val_size]
    y_train_fit = y_train_processed[:-val_size]
    
    history = model.fit(
        datagen.flow(X_train_fit, y_train_fit, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        steps_per_epoch=len(X_train_fit) // batch_size,
        verbose=1
    )
else:
    history = model.fit(
        X_train_processed, y_train_processed,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )

print("\nEvaluating model on test set...")
test_loss, test_accuracy = model.evaluate(X_test_processed, y_test_processed, verbose=0)

# Additional metrics
y_pred = np.argmax(model.predict(X_test_processed), axis=1)
y_true = y_test.flatten()

metrics = {
    'timestamp': datetime.now().isoformat(),
    'epochs_trained': len(history.history['loss']),
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'precision': float(precision_score(y_true, y_pred, average='weighted')),
    'recall': float(recall_score(y_true, y_pred, average='weighted')),
    'f1_score': float(f1_score(y_true, y_pred, average='weighted')),
    'final_train_accuracy': float(history.history['accuracy'][-1]),
    'final_val_accuracy': float(history.history['val_accuracy'][-1])
}

# Save metrics
os.makedirs('logs', exist_ok=True)
with open('logs/training_logs.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Save model
save_model(model, 'models/cifar10_model.h5')

print("\n✅ Training complete! Model saved successfully.")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

return model, history
```

def retrain_model(new_data_path, epochs=20, batch_size=128):
"""
Retrain existing CIFAR-10 model with new images.

```
Args:
    new_data_path: Directory containing new class subfolders with images
    epochs: Number of retraining epochs
    batch_size: Training batch size

Returns:
    model: Retrained Keras model
"""
from src.model import load_model
from glob import glob
from PIL import Image

print("Loading existing model...")
model = load_model('models/cifar10_model.h5')

print(f"Loading new data from {new_data_path}...")
new_images, new_labels = [], []

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

if os.path.exists(new_data_path):
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(new_data_path, class_name)
        if os.path.exists(class_dir):
            image_files = glob(os.path.join(class_dir, '*.jpg')) + \
                          glob(os.path.join(class_dir, '*.png'))
            for img_path in image_files:
                img = Image.open(img_path).resize((32, 32)).convert('RGB')
                new_images.append(np.array(img))
                new_labels.append(class_idx)

if new_images:
    print(f"Found {len(new_images)} new images for retraining.")
    new_images = np.array(new_images)
    new_labels = np.array(new_labels).reshape(-1, 1)
    # Load original dataset for mix
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    sample_size = min(10000, len(X_train))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_combined = np.concatenate([X_train[indices], new_images])
    y_combined = np.concatenate([y_train[indices], new_labels])
else:
    print("No new images found. Retraining on original dataset.")
    (X_combined, y_combined), (X_test, y_test) = cifar10.load_data()

# Preprocess
X_processed, y_processed = preprocess_data(X_combined, y_combined)
X_test_processed, y_test_processed = preprocess_data(X_test, y_test)

callbacks = get_training_callbacks()

history = model.fit(
    X_processed, y_processed,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test_processed, y_test_processed, verbose=0)

print(f"\n✅ Retraining complete! New Test Accuracy: {test_accuracy:.4f}")

# Save retrained model
save_model(model, 'models/cifar10_model.h5')

# Update retrain logs
metrics = {
    'timestamp': datetime.now().isoformat(),
    'retrain': True,
    'new_samples_added': len(new_images) if new_images else 0,
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss)
}

os.makedirs('logs', exist_ok=True)
with open('logs/retrain_logs.json', 'a') as f:
    json.dump(metrics, f, indent=4)
    f.write('\n')

return model
```

if **name** == '**main**':
# Entry point: train model from scratch
model, history = train_model(epochs=50)

