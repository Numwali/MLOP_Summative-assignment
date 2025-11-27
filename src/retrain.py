"""
retrain.py
Loads new images under data/retrain, backs up latest model, fine-tunes and saves a new model.
"""
import os
import shutil
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks
from .preprocessing import preprocess_arrays, load_cifar10, preprocess_single_image_from_file


DEFAULT_LATEST = os.path.join('models', 'cifar10_model_latest.keras')


def load_images_from_retrain_dir(retrain_dir='data/retrain', target_size=(32,32)):
    Xr, yr = [], []
    if not os.path.exists(retrain_dir):
        return np.empty((0,)+target_size+(3,)), np.array([])
    for class_entry in sorted(os.listdir(retrain_dir)):
        class_path = os.path.join(retrain_dir, class_entry)
        if not os.path.isdir(class_path):
            continue
        # Resolve class index
        if class_entry.isdigit():
            class_idx = int(class_entry)
        else:
            # try to read classes.json if present
            classes_json = os.path.join('models', 'classes.json')
            if os.path.exists(classes_json):
                import json
                with open(classes_json, 'r') as f:
                    classes = json.load(f)
                if class_entry in classes:
                    class_idx = classes.index(class_entry)
                else:
                    print(f"[retrain] Unknown class folder '{class_entry}', skipping.")
                    continue
            else:
                print(f"[retrain] classes.json not found and folder '{class_entry}' is not numeric. Skipping.")
                continue
        for fname in sorted(os.listdir(class_path)):
            if not fname.lower().endswith(('.png','.jpg','.jpeg')):
                continue
            fpath = os.path.join(class_path, fname)
            try:
                arr = preprocess_single_image_from_file(fpath, target_size=target_size)
                Xr.append(arr)
                yr.append(class_idx)
            except Exception as e:
                print(f"[retrain] Failed to load {fpath}: {e}")
    if len(Xr) == 0:
        return np.empty((0,)+target_size+(3,)), np.array([])
    Xr = np.vstack([x[np.newaxis, ...] if x.ndim==3 else x for x in Xr])
    yr = np.array(yr)
    return Xr, yr


def retrain_from_directory(retrain_dir='data/retrain', epochs=3, batch_size=64, backup=True):
    Xr, yr = load_images_from_retrain_dir(retrain_dir)
    if Xr.size == 0:
        print('[retrain] No retrain data found.')
        return None
    # Convert to proper shapes and preprocess
    # Currently Xr is batch x H x W x C (uint8)
    Xr = Xr.astype('float32') / 255.0
    from tensorflow.keras.utils import to_categorical
    classes_json = os.path.join('models', 'classes.json')
    if os.path.exists(classes_json):
        import json
        with open(classes_json, 'r') as f:
            classes = json.load(f)
        num_classes = len(classes)
    else:
        num_classes = 10

    yr_cat = to_categorical(yr, num_classes)

    latest = DEFAULT_LATEST
    if not os.path.exists(latest):
        raise FileNotFoundError('[retrain] Latest model not found. Train first.')
    if backup:
        backup_path = f"models/cifar10_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
        shutil.copy2(latest, backup_path)
        print(f"[retrain] Backed up latest model to {backup_path}")

    model = load_model(latest)
    # compile with a lower LR for fine-tuning
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-7)
    ]

    history = model.fit(Xr, yr_cat, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=cb)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    newpath = os.path.join('models', f'cifar10_model_retrained_{ts}.keras')
    model.save(newpath)
    model.save(latest)
    print(f"[retrain] Retrained model saved to {newpath} and overwritten {latest}")

    # save retrain log summary
    retrain_log = {
        'timestamp': ts,
        'retrain_dir': retrain_dir,
        'new_model': newpath,
        'history': {k: [float(x) for x in v] for k, v in history.history.items()}
    }
    import json
    with open(os.path.join('logs', f'retrain_log_{ts}.json'), 'w') as f:
        json.dump(retrain_log, f, indent=2)
    print('[retrain] retrain log saved to logs/')
    return retrain_log
