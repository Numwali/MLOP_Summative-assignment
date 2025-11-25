from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.model import load_model, create_cifar10_model, compile_model, save_model
from src.preprocessing import preprocess_single_image, preprocess_data

from tensorflow.keras.datasets import cifar10
import numpy as np
from PIL import Image
import io
import time
import os
import logging

# ------------------------------------
# Logging
# ------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------
# FastAPI App
# ------------------------------------
app = FastAPI(title="CIFAR-10 Dashboard API")

# Enable all CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track uptime
start_time = time.time()

# Serve dashboard static files
app.mount("/web", StaticFiles(directory="web"), name="web")

# ------------------------------------
# Model Setup
# ------------------------------------
MODEL_DIR = "models"
LATEST_MODEL_FILENAME = "cifar10_model_latest.keras"
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, LATEST_MODEL_FILENAME)

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Load or create model
if os.path.exists(model_path):
    logger.info("Loading existing model...")
    model = load_model(model_path)
else:
    logger.info("No saved model found. Creating a new model...")
    model = create_cifar10_model()
    model = compile_model(model)
    save_model(model, model_path)
    logger.info(f"Model saved: {model_path}")

# ------------------------------------
# ROUTES
# ------------------------------------

@app.get("/", response_class=HTMLResponse)
def read_dashboard():
    html_file = os.path.join("web", "dashboard.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)


# ------------------------------------
# Prediction Route
# ------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        x = preprocess_single_image(img_bytes)

        preds = model.predict(x)
        class_index = int(np.argmax(preds))

        return {
            "class_index": class_index,
            "class_name": class_names[class_index],
            "confidence": float(np.max(preds)),
        }

    except Exception as e:
        logger.exception("Prediction failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------
# Retraining Route
# ------------------------------------
@app.post("/retrain")
async def retrain(files: list[UploadFile] = File(...)):
    try:
        new_images = []
        new_labels = []

        # Read uploaded images
        for file in files:
            filename = file.filename

            # First folder name is the class
            class_name = filename.split("/")[0]

            if class_name in class_names:
                class_idx = class_names.index(class_name)

                img_bytes = await file.read()
                img = Image.open(io.BytesIO(img_bytes)).resize((32, 32)).convert("RGB")

                new_images.append(np.array(img))
                new_labels.append(class_idx)

        if not new_images:
            return JSONResponse({"error": "No valid images found"}, status_code=400)

        # Load CIFAR-10 dataset
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        # Combine with user data
        X_combined = np.concatenate([X_train[:10000], np.array(new_images)])
        y_combined = np.concatenate([y_train[:10000], np.array(new_labels).reshape(-1, 1)])

        # Preprocess
        X_processed, y_processed = preprocess_data(X_combined, y_combined)
        X_test_processed, y_test_processed = preprocess_data(X_test, y_test)

        # Retrain
        model.fit(
            X_processed, y_processed,
            epochs=5,
            validation_split=0.2,
            verbose=1
        )

        # Save updated model
        save_model(model, model_path)

        # Evaluate new performance
        test_loss, test_acc = model.evaluate(X_test_processed, y_test_processed, verbose=0)

        return {
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "message": "Model retrained and saved successfully."
        }

    except Exception as e:
        logger.exception("Retraining failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------
# Uptime Route
# ------------------------------------
@app.get("/uptime")
def uptime():
    elapsed = int(time.time() - start_time)
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    return {"uptime": f"{hours}h {minutes}m {seconds}s"}


# ------------------------------------
# Run Server
# ------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

