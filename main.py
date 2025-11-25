"""
main.py

FastAPI backend for CIFAR-10 model inference and training.
"""

import os
import logging
import numpy as np
from fastapi import FastAPI, UploadFile, File

from src.model import (
    create_cifar10_model,
    compile_model,
    save_model,
    load_model,
    MODELS_DIR,
    LATEST_MODEL,
)
from src.preprocessing import preprocess_single_image

# -------------------------
# Logger setup
# -------------------------

logger = logging.getLogger("main")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(title="CIFAR-10 API", version="1.0")

MODEL_PATH = os.path.join(MODELS_DIR, LATEST_MODEL)

# -------------------------
# Load or Create Model
# -------------------------

if os.path.exists(MODEL_PATH):
    logger.info("Loading existing model...")
    model = load_model(MODEL_PATH)
else:
    logger.info("No saved model found. Creating a new model...")
    model = create_cifar10_model()
    compile_model(model)
    save_model(model)

# -------------------------
# Routes
# -------------------------

@app.get("/")
def index():
    return {"message": "CIFAR-10 API running successfully."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    x = preprocess_single_image(img_bytes)

    preds = model.predict(x)
    pred_class = int(np.argmax(preds))

    return {
        "predicted_class": pred_class,
        "confidence": float(np.max(preds))
    }


@app.get("/model-info")
def get_model_info():
    return {"model_path": MODEL_PATH, "exists": os.path.exists(MODEL_PATH)}

