import os
import json
import numpy as np
from PIL import Image
from src.prediction import class_names

def load_classes(path="models/classes.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            classes = json.load(f)
        return classes
    return class_names

def prepare_image(image: Image.Image, target_size=(32,32)):
    img = image.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

