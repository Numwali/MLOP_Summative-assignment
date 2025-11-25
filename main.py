"""
main.py

FastAPI server for the CIFAR-10 project:
- Serves dashboard (web/dashboard.html)
- Exposes /predict, /retrain, /metrics, /uptime
- Serves visual assets from notebook/logs via /assets
"""

import os
import time
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.model import load_model, create_cifar10_model, compile_model, save_model, get_training_callbacks
from src.preprocessing import preprocess_single_image, preprocess_data, load_images_from_directory

# ---------- App init ----------
app = FastAPI(title="CIFAR-10 API & Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Uptime
start_time = time.time()

# Mount the dashboard folder
if not os.path.exists("web"):
    os.makedirs("web", exist_ok=True)

app.mount("/web", StaticFiles(directory="web"), name="web")

# Also expose notebook logs visuals (so dashboard can show confusion matrix, etc.)
if os.path.exists("notebook/logs"):
    app.mount("/assets", StaticFiles(directory="notebook/logs"), name="assets")
else:
    # create the directory so mounting doesn't fail later
    os.makedirs("notebook/logs", exist_ok=True)
    app.mount("/assets", StaticFiles(directory="notebook/logs"), name="assets")

# Expose models directory (so you can inspect saved models from the browser if desired)
os.makedirs("models", exist_ok=True)
app.mount("/models", StaticFiles(directory="models"), name="models")


# ---------- Model load/create ----------
MODEL_LATEST = os.path.join("models", "cifar10_model_latest.keras")
if os.path.exists(MODEL_LATEST):
    try:
        model = load_model(MODEL_LATEST)
    except Exception as e:
        # If load fails, create fresh model
        model = create_cifar10_model()
        model = compile_model(model)
        save_model(model, name="cifar10_model_latest.keras")
else:
    # No model on disk: create, compile and save an initial weights file
    model = create_cifar10_model()
    model = compile_model(model)
    save_model(model, name="cifar10_model_latest.keras")

# CIFAR-10 labels
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ---------- Endpoints ----------

@app.get("/", response_class=HTMLResponse)
def dashboard():
    """
    Serve the dashboard HTML (web/dashboard.html).
    """
    index_path = os.path.join("web", "dashboard.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h3>Dashboard not found. Put web/dashboard.html in the web/ folder.</h3>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Predict a single uploaded image.
    """
    try:
        contents = await image.read()
        arr = preprocess_single_image(contents)   # returns (1,32,32,3) normalized
        preds = model.predict(arr, verbose=0)
        idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][idx])
        return {
            "predicted_class": CLASS_NAMES[idx],
            "class_index": idx,
            "confidence": confidence,
            "all_probs": {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
async def retrain(files: list[UploadFile] = File(...)):
    """
    Receive multiple image files for retraining.
    Expects filenames in the form: <class_name>/<imagefile.jpg>
    or form field 'class_name' may be used on each file's filename.
    This endpoint will:
      - load uploaded images
      - combine with a sampled portion of CIFAR-10 train set
      - preprocess and train for a small number of epochs
      - save new latest model and write metrics to notebook/logs/training_logs.json
    """
    # collect uploaded images
    new_images = []
    new_labels = []

    for file in files:
        fn = file.filename or ""
        parts = fn.split("/")
        class_guess = parts[0] if parts else ""
        try:
            contents = await file.read()
            img = Image.open(io := (None)).close()  # dummy to satisfy linters (we won't use this line)
        except Exception:
            # read via PIL directly from bytes
            try:
                contents = await file.read()
            except Exception:
                contents = b""

        # fallback: determine class from filename (class/image.jpg) else skip
        if "/" in file.filename:
            class_name = file.filename.split("/")[0]
        else:
            # if filename contains underscore label like cat_123.jpg, try to find class token
            class_name = None
            for cname in CLASS_NAMES:
                if cname in file.filename.lower():
                    class_name = cname
                    break

        if class_name not in CLASS_NAMES:
            # skip file (require class folder or class in filename)
            continue

        # read file contents again
        try:
            await file.seek(0)
            raw = await file.read()
            pil = Image.open(BytesIO(raw)).resize((32, 32)).convert("RGB")
            new_images.append(np.array(pil))
            new_labels.append(CLASS_NAMES.index(class_name))
        except Exception:
            # try one more time reading via PIL from previously read bytes
            try:
                pil = Image.open(io.BytesIO(contents)).resize((32, 32)).convert("RGB")
                new_images.append(np.array(pil))
                new_labels.append(CLASS_NAMES.index(class_name))
            except Exception:
                continue

    if len(new_images) == 0:
        return JSONResponse({"error": "No valid labeled images found in upload. Use folder names like cat/img1.jpg"}, status_code=400)

    # Combine with CIFAR-10 subset
    from tensorflow.keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    sample_size = min(10000, len(X_train))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_combined = np.concatenate([X_train[indices], np.array(new_images)])
    y_combined = np.concatenate([y_train[indices], np.array(new_labels).reshape(-1, 1)])

    # Preprocess
    X_processed, y_processed = preprocess_data(X_combined, y_combined)
    X_test_processed, y_test_processed = preprocess_data(X_test, y_test)

    # Train for a small number of epochs
    callbacks = get_training_callbacks()
    history = model.fit(X_processed, y_processed, batch_size=128, epochs=3, validation_split=0.1, callbacks=callbacks, verbose=1)

    # Save new model
    save_model(model, name=f"cifar10_model_retrained_{int(time.time())}.keras")

    # Evaluate and write logs
    test_loss, test_acc = model.evaluate(X_test_processed, y_test_processed, verbose=0)
    training_metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs_trained": len(history.history["loss"]),
        "history": {k: v for k, v in history.history.items()}
    }

    os.makedirs("notebook/logs", exist_ok=True)
    with open("notebook/logs/training_logs.json", "w") as f:
        json.dump(training_metrics, f, indent=2)

    return {"message": "Retrain complete", "test_accuracy": float(test_acc), "test_loss": float(test_loss)}


@app.get("/metrics")
def metrics():
    p = "notebook/logs/training_logs.json"
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return {"error": "No metrics found. Train model first."}


@app.get("/uptime")
def uptime():
    elapsed = int(time.time() - start_time)
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    return {"uptime": f"{hours}h {minutes}m {seconds}s"}


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

