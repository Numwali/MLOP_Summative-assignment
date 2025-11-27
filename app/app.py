from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.prediction import load_latest_model, predict_image
from src.retrain import retrain_from_directory
from app.utils import prepare_image
from PIL import Image
import io
import os

app = FastAPI(
    title="CIFAR-10 ML API",
    description="API for CIFAR-10 image classification with retraining capabilities",
    version="1.0"
)

# Allow CORS (so UI can call API from different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model at startup
model = None
@app.on_event("startup")
def load_model_on_startup():
    global model
    model = load_latest_model()
    print("Model loaded successfully.")

# --- Health Check ---
@app.get("/uptime")
def uptime():
    try:
        _ = load_latest_model()
        return {"status": "ok", "message": "Model is loaded and ready!"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# --- Predict single image ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
        x = prepare_image(img)
        result = predict_image(model, x[0])
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# --- Predict batch images ---
@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            img_bytes = await file.read()
            img = Image.open(io.BytesIO(img_bytes))
            x = prepare_image(img)
            result = predict_image(model, x[0])
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})
    return {"batch_results": results}

# --- Trigger retraining ---
@app.post("/retrain")
def retrain():
    retrain_dir = "data/retrain"
    if not os.path.exists(retrain_dir):
        return JSONResponse(status_code=400, content={"status": "error", "message": f"No retrain data in {retrain_dir}"})
    try:
        retrain_log = retrain_from_directory(retrain_dir)
        # Reload model after retraining
        global model
        model = load_latest_model()
        return {"status": "ok", "message": "Retraining completed", "retrain_log": retrain_log}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

