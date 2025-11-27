# src/prediction.py

import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# CIFAR-10 class labels
class_names = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def load_latest_model(path="models/cifar10_model_latest.keras"):
    """Load the most recently trained CIFAR-10 model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return load_model(path)


def preprocess_single_image(pil_img, target_size=(32,32)):
    """Convert PIL → normalized numpy array (HxWxC)."""
    img = pil_img.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return arr


def preprocess_for_predict(arr):
    """Add batch dimension."""
    x = np.expand_dims(arr, axis=0)
    return x


def decode_predictions(probs, top_k=3):
    """Return top-k class names + probabilities."""
    top = probs.argsort()[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in top]


def predict_image(model, arr, top_k=3):
    """
    Given a normalized (32x32x3) array → return predictions.
    """
    x = preprocess_for_predict(arr)
    probs = model.predict(x)[0]

    top3 = decode_predictions(probs, top_k)
    idx = int(np.argmax(probs))

    return {
        "pred_class": class_names[idx],
        "confidence": float(probs[idx]),
        "top3": top3
    }


def predict_single_image(model, pil_image):
    """
    Accepts a PIL.Image and returns (label, confidence).
    """
    arr = preprocess_single_image(pil_image)
    res = predict_image(model, arr)
    return res["pred_class"], res["confidence"]
