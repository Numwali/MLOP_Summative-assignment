# main.py
"""
FastAPI app to serve the CIFAR-10 dashboard and API endpoints.

Place this file in the repo root (same level as the `web/` folder).
"""

import os
import time
import json
from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from PIL import Image

# Import your project utilities (ensure src is a package or adjust import path)
from src.model import load_model, create_cifar10_model, compile_model, save_model
from src.preprocessing import preprocess_single_image, normalize_image_array

# App setup
app = FastAPI(title="CIFAR-10 Dashboard API")

# Allow frontend access from anywhere for demo (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the web folder at root (so visiting / returns index.html)
# StaticFiles with html=True will serve index.html automatically at root.
web_dir = "web"
if not os.path.isdir(web_dir):
    raise RuntimeError(f"Missing web/ directory. Create {web_dir}/ and add index.html, dashboard.js, style.css")

app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

# Model configuration
MODELS_DIR = "models"
LATEST_MODEL = os.path.join(MODELS_DIR, "cifar10_model_latest.keras")

os.makedirs(MODELS_DIR, exist_ok=True)

# Try to load model; if missing create a fresh model and save it
try:
    if os.path.exists(LATEST_MODEL):
        model = load_model(LATEST_MODEL)
    else:
        # create/compile/save a fresh model (lightweight)
        model = create_cifar10_model()
        model = compile_model(model)
        # save as latest (so subsequent runs load it)
        save_model(model, name=os.path.basename(LATEST_MODEL), models_dir=MODELS_DIR, make_latest=True)
except Exception as e:
    # If something goes wrong, create a new compiled model instance but don't crash the server
    print("Warning: failed to load existing model - creating a fresh model.", e)
    model = create_cifar10_model()
    model = compile_model(model)

# CIFAR-10 labels (keeps consistent with your training)
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Track uptime
start_time = time.time()


# -------- API endpoints used by the dashboard --------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a single image file in form field 'file' and return prediction.
    Frontend should send FormData with key 'file'.
    """
    try:
        contents = await file.read()
        x = preprocess_single_image(contents)  # returns shape (1,32,32,3) normalized
        preds = model.predict(x, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        top_conf = float(preds[top_idx])
        # top 3
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
    Accept multiple files as a retraining batch.
    Expect files to be uploaded with preserved folder names in the filename (e.g. 'cat/img1.jpg')
    Frontend should append all files under the 'files' key (FormData).
    """
    try:
        new_images = []
        new_labels = []
        for f in files:
            filename = f.filename  # if browser preserves path, may be like 'cat/image.jpg'
            # class detection: take the first path component
            class_name = filename.split("/")[-2] if "/" in filename else filename.split("\\")[-2] if "\\" in filename else None
            # fallback: if user names files like cat_1.jpg, attempt to infer
            if not class_name:
                # unable to infer class -> skip
                continue
            if class_name not in CLASS_NAMES:
                # not a valid class, skip
                continue

            contents = await f.read()
            img = Image.open(BytesIO(contents)).resize((32, 32)).convert("RGB")
            new_images.append(np.array(img))
            new_labels.append(CLASS_NAMES.index(class_name))

        if len(new_images) == 0:
            return JSONResponse({"error": "No valid labeled files found for retraining. Use folder structure <class_name>/image.jpg"}, status_code=400)

        # Combine with a small sampled portion of CIFAR-10 for stability (optional)
        from tensorflow.keras.datasets import cifar10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        sample_size = min(10000, len(X_train))
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_combined = np.concatenate([X_train[indices], np.array(new_images)])
        y_combined = np.concatenate([y_train[indices], np.array(new_labels).reshape(-1, 1)])

        # Preprocess using your preprocessing utilities (normalize + one-hot)
        from src.preprocessing import preprocess_data
        X_p, y_p = preprocess_data(X_combined, y_combined)
        X_test_p, y_test_p = preprocess_data(X_test, y_test)

        # Retrain (short)
        model.fit(X_p, y_p, batch_size=128, epochs=3, validation_split=0.1, verbose=1)
        # Save updated model as latest
        save_model(model, name=os.path.basename(LATEST_MODEL), models_dir=MODELS_DIR, make_latest=True)

        loss, acc = model.evaluate(X_test_p, y_test_p, verbose=0)
        return {"test_loss": float(loss), "test_accuracy": float(acc)}
    except Exception as e:
        return JSONResponse({"error": f"Retrain failed: {str(e)}"}, status_code=500)


@app.get("/metrics")
def get_metrics():
    """Return training logs if present (expects logs/training_logs.json written by training)"""
    log_path = "logs/training_logs.json"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            metrics = json.load(f)
        return metrics
    # If there's a notebook logs directory (common during development), try that
    alt = "notebook/logs/training_logs.json"
    if os.path.exists(alt):
        with open(alt, "r") as f:
            metrics = json.load(f)
        return metrics
    return {"message": "No metrics found. Run training to produce logs."}


@app.get("/uptime")
def uptime():
    elapsed = int(time.time() - start_time)
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    return {"uptime": f"{hours}h {minutes}m {seconds}s"}


# Helper to return index explicitly (optional, static mount already serves index.html)
@app.get("/index")
def index():
    index_path = os.path.join(web_dir, "index.html")
    return FileResponse(index_path)

