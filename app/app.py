# app/app.py
import io
import os
import time
import zipfile
import shutil
import json
from typing import Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from src.prediction import load_latest_model, predict_single_image, preprocess_single_image
from src.retrain import retrain_from_directory, evaluate_model_on_testset

START_TIME = time.time()
MODEL_PATH = "models/cifar10_model_latest.keras"
DATA_RETRAIN_DIR = "data/train"   # Zip extracts here
METRICS_LOG = "logs/eval_metrics.json"

app = FastAPI(title="CIFAR-10 ML API (Full)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# global model
model = None

@app.on_event("startup")
def startup_event():
    global model
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    # Try load model
    try:
        model = load_latest_model(MODEL_PATH)
        app.state.model_loaded = True
        print("Model loaded at startup.")
    except Exception as e:
        app.state.model_loaded = False
        print("No model loaded at startup:", e)


@app.get("/")
def root():
    return {"message": "CIFAR-10 API running", "model_loaded": app.state.model_loaded}


@app.get("/uptime/")
def uptime():
    return {"uptime_seconds": int(time.time() - START_TIME)}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Predict single uploaded image."""
    global model
    if not app.state.model_loaded:
        raise HTTPException(status_code=500, detail="No model loaded.")

    data = await file.read()
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    label, confidence = predict_single_image(model, pil)
    return {"class": label, "confidence": float(confidence)}


@app.post("/upload-data/")
async def upload_data(file: UploadFile = File(...)):
    """Upload a zip file containing class folders (ex: airplane/, cat/, ...). Extracts to data/train/"""
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file")

    data = await file.read()
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(DATA_RETRAIN_DIR)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Bad zip file")

    return {"message": "Dataset uploaded and extracted", "extracted_to": DATA_RETRAIN_DIR}


@app.post("/retrain/")
def retrain(epochs: int = 3, batch_size: int = 32):
    """Trigger retraining from images in data/train/; returns training history and re-evaluation metrics."""
    global model
    # Check retrain data
    if not any(os.scandir(DATA_RETRAIN_DIR)):
        return JSONResponse(status_code=400, content={"message": "No retrain data found in data/train/ (extract zip first)"})

    # Backup current model
    if os.path.exists(MODEL_PATH):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = f"models/cifar10_model_backup_{timestamp}.keras"
        shutil.copy2(MODEL_PATH, backup_path)

    # call retrain routine (this returns retrain_log with history)
    try:
        retrain_log = retrain_from_directory(retrain_dir=DATA_RETRAIN_DIR, epochs=epochs, batch_size=batch_size, backup=False, latest_model_path=MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {e}")

    # reload model into memory
    try:
        model = load_latest_model(MODEL_PATH)
        app.state.model_loaded = True
    except Exception as e:
        app.state.model_loaded = False
        print("Failed to reload model:", e)

    # evaluate model on CIFAR-10 test set for metrics
    try:
        metrics = evaluate_model_on_testset(MODEL_PATH)
        # save metrics to logs
        with open(METRICS_LOG, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        metrics = {"eval_error": str(e)}

    return {"retrain_log": retrain_log, "eval_metrics": metrics}


@app.get("/metrics/")
def get_metrics():
    """Return last evaluation metrics and simple class distribution from retrain folder"""
    metrics = {}
    if os.path.exists(METRICS_LOG):
        with open(METRICS_LOG, "r") as f:
            metrics = json.load(f)
    # class distribution if any retrain data
    dist = {}
    if os.path.isdir(DATA_RETRAIN_DIR):
        for d in sorted(os.listdir(DATA_RETRAIN_DIR)):
            p = os.path.join(DATA_RETRAIN_DIR, d)
            if os.path.isdir(p):
                dist[d] = len([f for f in os.listdir(p) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    metrics["retrain_class_distribution"] = dist
    return metrics
