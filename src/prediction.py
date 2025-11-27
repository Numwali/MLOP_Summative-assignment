"""
prediction.py
Utilities to load model and run single-image / batch predictions.
"""
import os
import io
import json
from typing import List, Tuple
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


DEFAULT_LATEST = os.path.join('models', 'cifar10_model_latest.keras')
CLASSES_PATH = os.path.join('models', 'classes.json')


class Predictor:
    def __init__(self, model_path: str = DEFAULT_LATEST, classes_path: str = CLASSES_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = load_model(model_path)
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                self.classes = json.load(f)
        else:
            # fallback to CIFAR-10 default order
            self.classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    def _prepare_pil(self, pil_img: Image.Image, target_size=(32,32)) -> np.ndarray:
        pil = pil_img.convert('RGB').resize(target_size)
        arr = np.array(pil).astype('float32') / 255.0
        return np.expand_dims(arr, 0)

    def predict_from_pil(self, pil_img: Image.Image, top_k: int = 3):
        x = self._prepare_pil(pil_img)
        probs = self.model.predict(x)[0]
        return self._format_probs(probs, top_k)

    def predict_from_bytes(self, img_bytes: bytes, top_k: int = 3):
        pil = Image.open(io.BytesIO(img_bytes))
        return self.predict_from_pil(pil, top_k)

    def predict_from_file(self, path: str, top_k: int = 3):
        pil = Image.open(path)
        return self.predict_from_pil(pil, top_k)

    def _format_probs(self, probs: np.ndarray, top_k: int):
        idxs = probs.argsort()[::-1][:top_k]
        results = []
        for i in idxs:
            results.append({
                'class_index': int(i),
                'label': self.classes[i],
                'confidence': float(probs[i])
            })
        predicted = results[0]
        return {'predicted': predicted, 'top_k': results}

