import uvicorn
import time
import zipfile
import io
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from src.prediction import predict_single_image
from src.preprocessing import preprocess_single_image
from src.retrain import retrain_model, generate_gradcam, extract_filters, get_confused_samples
from src.model import load_model
import shutil
import numpy as np
from PIL import Image
import json

# -----------------------------------------------------
# GLOBAL SETTINGS
# -----------------------------------------------------
START_TIME = time.time()
MODEL_PATH = "models/cifar10_model_latest.keras"
DATASET_TRAIN_PATH = "data/train/"
METRICS_FILE = "logs/eval_metrics.json"

# -----------------------------------------------------
# INIT FASTAPI APP
# -----------------------------------------------------
app = FastAPI(
    title="CIFAR-10 Image Classification API",
    version="1.0",
    description="End-to-End ML System: Prediction, Retraining, Monitoring"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model once at startup
model = load_model(MODEL_PATH)


# -----------------------------------------------------
# ROOT
# -----------------------------------------------------
@app.get("/")
def home():
    return {"message": "CIFAR-10 API running successfully!"}


# -----------------------------------------------------
# PREDICTION ENDPOINT
# -----------------------------------------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))

    pred_class, confidence = predict_single_image(model, img)

    return {
        "class": pred_class,
        "confidence": float(confidence)
    }


# -----------------------------------------------------
# UPLOAD ZIP DATA FOR RETRAINING
# -----------------------------------------------------
@app.post("/upload-data/")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        return {"error": "Please upload a ZIP file."}

    # Extract ZIP into data/train/
    zip_bytes = await file.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_ref:
        zip_ref.extractall(DATASET_TRAIN_PATH)

    return {"message": "Dataset uploaded and extracted successfully!"}


# -----------------------------------------------------
# RETRAINING ENDPOINT
# -----------------------------------------------------
@app.post("/retrain/")
async def retrain():

    history, new_model = retrain_model(
        existing_model=model,
        data_path=DATASET_TRAIN_PATH,
        save_path=MODEL_PATH
    )

    # Save updated metrics
    metrics = {
        "accuracy_curve": history.history["accuracy"],
        "loss_curve": history.history["loss"]
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f)

    # Replace global model
    global model
    model = load_model(MODEL_PATH)

    return metrics


# -----------------------------------------------------
# METRICS + EXPLAINABILITY
# -----------------------------------------------------
@app.get("/metrics/")
def get_metrics():

    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            metrics_data = json.load(f)
    else:
        metrics_data = {"accuracy": 0, "loss": 0}

    # GradCAM, Filters, Confusion Samples
    gradcam_img = generate_gradcam(model)
    filters_img = extract_filters(model)
    confused_img = get_confused_samples(model)

    # Convert to base64 (for Streamlit)
    def img_to_base64(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "accuracy": float(metrics_data.get("accuracy", 0)),
        "loss": float(metrics_data.get("loss", 0)),
        "accuracy_curve": metrics_data.get("accuracy_curve", []),
        "loss_curve": metrics_data.get("loss_curve", []),
        "class_distribution": {
            cls: len(os.listdir(os.path.join(DATASET_TRAIN_PATH, cls)))
            for cls in os.listdir(DATASET_TRAIN_PATH)
            if os.path.isdir(os.path.join(DATASET_TRAIN_PATH, cls))
        },
        "gradcam": img_to_base64(gradcam_img),
        "filters": img_to_base64(filters_img),
        "confused": img_to_base64(confused_img)
    }


# -----------------------------------------------------
# SYSTEM UPTIME
# -----------------------------------------------------
@app.get("/uptime/")
def uptime():
    return {"uptime": time.time() - START_TIME}


# -----------------------------------------------------
# RUN
# -----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

