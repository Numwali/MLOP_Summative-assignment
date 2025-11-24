"""
src/prediction.py

Prediction utilities for CIFAR-10.

This module provides:
- CIFAR10Predictor: class to handle model loading and predictions
- Single image prediction with top-3 probabilities
- Batch prediction
- Utility function to quickly predict a single image without instantiating the class
"""

import numpy as np
from typing import Dict, List, Any, Optional

from src.preprocessing import preprocess_single_image, decode_prediction, CLASS_NAMES
from src.model import load_model


class CIFAR10Predictor:
    """
    Predictor class for CIFAR-10 CNN model.

    Attributes:
        model_path (str): path to saved Keras model
        model (keras.Model): loaded Keras model
        class_names (list): list of class names
    """

    def __init__(self, model_path: str = "models/cifar10_model.h5"):
        """
        Initialize predictor and load model.

        Args:
            model_path: path to saved model
        """
        self.model_path = model_path
        self.model = load_model(model_path)
        self.class_names = CLASS_NAMES

    def predict_single(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict the class of a single image.

        Args:
            image_bytes: image file in bytes

        Returns:
            dict: {
                'predicted_class': str,
                'confidence': float,
                'top_3_predictions': list of dicts,
                'all_probabilities': dict
            }
        """
        # Preprocess image
        img_array = preprocess_single_image(image_bytes)

        # Predict
        predictions = self.model.predict(img_array, verbose=0)[0]

        # Decode main prediction
        class_name, confidence = decode_prediction(predictions)

        # Top-3 predictions
        top_3_idx = np.argsort(predictions)[-3:][::-1]
        top_3_predictions = [
            {"class": self.class_names[i], "confidence": float(predictions[i])}
            for i in top_3_idx
        ]

        all_probabilities = {
            self.class_names[i]: float(predictions[i]) for i in range(len(self.class_names))
        }

        return {
            "predicted_class": class_name,
            "confidence": confidence,
            "top_3_predictions": top_3_predictions,
            "all_probabilities": all_probabilities,
        }

    def predict_batch(self, images_array: np.ndarray) -> List[Dict[str, Any]]:
        """
        Predict classes for a batch of preprocessed images.

        Args:
            images_array: numpy array of shape (batch_size, 32, 32, 3)

        Returns:
            List of dictionaries with predicted class and confidence
        """
        predictions = self.model.predict(images_array, verbose=0)
        results = []
        for pred in predictions:
            class_name, confidence = decode_prediction(pred)
            results.append({"predicted_class": class_name, "confidence": confidence})
        return results

    def reload_model(self, model_path: Optional[str] = None):
        """
        Reload the model, useful after retraining.

        Args:
            model_path: path to model; if None, reloads the previous model
        """
        if model_path:
            self.model_path = model_path
        self.model = load_model(self.model_path)


# ----------------------------
# Convenience function
# ----------------------------
def predict_image(image_bytes: bytes, model_path: str = "models/cifar10_model.h5") -> Dict[str, Any]:
    """
    Quick helper to predict a single image without instantiating CIFAR10Predictor.

    Args:
        image_bytes: image file in bytes
        model_path: path to saved model

    Returns:
        Prediction dictionary (same format as predict_single)
    """
    predictor = CIFAR10Predictor(model_path)
    return predictor.predict_single(image_bytes)


# ----------------------------
# Simple CLI test
# ----------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        print("Usage: python prediction.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    predictor = CIFAR10Predictor()
    result = predictor.predict_single(image_bytes)
    print("\nPrediction Result:")
    print(result)

