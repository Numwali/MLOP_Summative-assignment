from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.model import load_model, create_cifar10_model, compile_model, save_model
from src.preprocessing import preprocess_data

from tensorflow.keras.datasets import cifar10
import numpy as np
import uvicorn
import time
import os
from PIL import Image
import json

# FastAPI app
app = FastAPI(title="CIFAR-10 Dashboard API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# uptime
start_time = time.time()

# static files
app.mount("/web", StaticFiles(directory="web"), name="web")

# model directory
model_dir = "models"
model_path = os.path.join(model_dir, "cifar10_model_latest.keras")
os.makedirs(model_dir, exist_ok=True)

# load or create model
if os.path.exists(model_path):
    print("Loading existing model...")
    model = load_model(model_path)
else:
    print("Creating new model...")
    model = create_cifar10_model()
    model = compile_model(model)
    save_model(model, model_path)

# CIFAR-10 class labels
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ---------------- ROOT ENDPOINT ----------------
@app.get("/", response_class=HTMLResponse)
def read_dashboard():
    with open("web/dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

# ---------------- PREDICT ENDPOINT ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).resize((32, 32)).convert("RGB")
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    class_index = int(np.argmax(preds))

    return {
        "class_index": class_index,
        "class_name": class_names[class_index]
    }

# ---------------- RETRAIN ENDPOINT ----------------
@app.post("/retrain")
async def retrain(files: list[UploadFile] = File(...)):
    new_images = []
    new_labels = []

    for file in files:
        filename = file.filename

        # extract label before the first underscore (e.g. "cat_01.jpg")
        class_name = filename.split("_")[0].lower()

        if class_name in class_names:
            class_idx = class_names.index(class_name)
            img = Image.open(file.file).resize((32, 32)).convert("RGB")

            new_images.append(np.array(img))
            new_labels.append(class_idx)

    if not new_images:
        return JSONResponse({"error": "No valid labeled images found"}, status_code=400)

    # combine with CIFAR10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # choose 10k samples to avoid slow training
    sample_size = min(10000, len(X_train))
    idx = np.random.choice(len(X_train), sample_size, replace=False)

    X_combined = np.concatenate([X_train[idx], np.array(new_images)])
    y_combined = np.concatenate([y_train[idx], np.array(new_labels).reshape(-1, 1)])

    # preprocess datasets
    X_processed, y_processed = preprocess_data(X_combined, y_combined)
    X_test_processed, y_test_processed = preprocess_data(X_test, y_test)

    # retrain
    model.fit(
        X_processed,
        y_processed,
        epochs=5,
        validation_split=0.2,
        verbose=1
    )

    # save updated model
    save_model(model, model_path)

    # evaluate
    loss, acc = model.evaluate(X_test_processed, y_test_processed, verbose=0)

    return {
        "test_accuracy": float(acc),
        "test_loss": float(loss)
    }

# ---------------- METRICS ----------------
@app.get("/metrics")
def get_metrics():
    if os.path.exists("logs/training_logs.json"):
        with open("logs/training_logs.json", "r") as f:
            return json.load(f)
    return {"error": "No metrics found"}

# ---------------- UPTIME ----------------
@app.get("/uptime")
def uptime():
    elapsed = int(time.time() - start_time)
    return {
        "uptime": f"{elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s"
    }

# run
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

