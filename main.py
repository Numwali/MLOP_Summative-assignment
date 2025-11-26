# main.py
"""
FastAPI app to serve the CIFAR-10 dashboard and API endpoints.

Place this file at the repo root (same level as the web/ folder).
"""

import os
import time
import json
from typing import List
from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from PIL import Image

# Import project utilities
from src.model import load_model, create_cifar10_model, compile_model, save_model
from src.preprocessing import preprocess_single_image, normalize_image_array


# ---------------------- APP SETUP ----------------------

app = FastAPI(title="CIFAR-10 Dashboard API")

# Allow frontend access from anywhere (for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the "web/" folder
web_dir = "web"
if not os.path.isdir(web_dir):
    raise RuntimeError("Missing web/ folder. Create web/ and add index.html, dashboard.js, style.css")

app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")


# ---------------------- MODEL HANDLING ----------------------

MODELS_DIR = "models"
LATEST_MODEL = os.path.join(MODELS_DIR, "cifar10_model_latest.keras")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load or create model
try:
    if os.path.exists(LATEST_MODEL):
        model = load_model(LATEST_MODEL)
    else:
        model = create_cifar10_model()
        model = compile_model(model)
        save_model(model, name=os.path.basename(LATEST_MODEL), models_dir=MODELS_DIR, make_latest=True)
except Exception as e:
    print("Warning: failed to load existing model. Creating a fresh model.", e)
    model = create_cifar10_model()
    model = compile_model(model)


# CIFAR-10 labels
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# uptime
start_time = time.time()


# ---------------------- API ENDPOINTS ----------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a single image under key 'file' and return prediction.
    """
    try:
        contents = await file.read()
        x = preprocess_single_image(contents)  # (1,32,32,3)
        preds = model.predict(x, verbose=0)[0]

        top_idx = int(np.argmax(preds))
        top_conf = float(preds[top_idx])
        top3 = np.argsort(preds)[-3:][::-1].tolist()

        top3_list = [{"class": CLASS_NAMES[i], "confidence": float(preds[i])} for i in top3]

        return {
            "predicted_class": CLASS_NAMES[top_idx],
            "class_index": top_idx,
            "confidence": top_conf,
            "top_3": top3_list,
            "all_probabilities": {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
        }

    except Exception as e:
        return JSONResponse({"error": f"Prediction failed: {str(e)}"}, status_code=500)


@app.post("/retrain")
async def retrain(files: List[UploadFile] = File(...)):
    """
    Retrain using uploaded batch of images.
    The uploaded files must preserve folder names: <class_name>/image.jpg
    """
    try:
        new_images = []
        new_labels = []

        for f in files:
            filename = f.filename

            # detect class from folder name
            class_name = (
                filename.split("/")[-2] if "/" in filename else
                filename.split("\\")[-2] if "\\" in filename else None
            )

            if not class_name or class_name not in CLASS_NAMES:
                continue

            contents = await f.read()
            img = Image.open(BytesIO(contents)).resize((32, 32)).convert("RGB")

            new_images.append(np.array(img))
            new_labels.append(CLASS_NAMES.index(class_name))

        if len(new_images) == 0:
            return JSONResponse(
                {"error": "No valid labeled files found. Use folder structure <class_name>/image.jpg"},
                status_code=400
            )

        # Mix with CIFAR-10 for stability
        from tensorflow.keras.datasets import cifar10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        sample_size = min(10000, len(X_train))
        indices = np.random.choice(len(X_train), sample_size, replace=False)

        X_combined = np.concatenate([X_train[indices], np.array(new_images)])
        y_combined = np.concatenate([y_train[indices], np.array(new_labels).reshape(-1, 1)])

        # Preprocess
        from src.preprocessing import preprocess_data
        X_train_p, y_train_p = preprocess_data(X_combined, y_combined)
        X_test_p, y_test_p = preprocess_data(X_test, y_test)

        # Retrain
        model.fit(X_train_p, y_train_p, epochs=3, batch_size=128, validation_split=0.1, verbose=1)

        # Save
        save_model(model, name=os.path.basename(LATEST_MODEL), models_dir=MODELS_DIR, make_latest=True)

        loss, acc = model.evaluate(X_test_p, y_test_p, verbose=0)

        return {"test_loss": float(loss), "test_accuracy": float(acc)}

    except Exception as e:
        return JSONResponse({"error": f"Retrain failed: {str(e)}"}, status_code=500)


@app.get("/metrics")
def get_metrics():
    """Return training logs if available."""
    log_path = "logs/training_logs.json"
    alt_path = "notebook/logs/training_logs.json"

    if os.path.exists(log_path):
        return json.load(open(log_path))

    if os.path.exists(alt_path):
        return json.load(open(alt_path))

    return {"message": "No metrics found. Run training to generate logs."}


@app.get("/uptime")
def uptime():
    elapsed = int(time.time() - start_time)
    h = elapsed // 3600
    m = (elapsed % 3600) // 60
    s = elapsed % 60
    return {"uptime": f"{h}h {m}m {s}s"}


@app.get("/index")
def index():
    return FileResponse(os.path.join(web_dir, "index.html"))
