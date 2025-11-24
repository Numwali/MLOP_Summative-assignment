from fastapi import FastAPI, UploadFile, File, Form
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

app = FastAPI(title="CIFAR-10 Dashboard API")

# Enable CORS for dashboard frontend

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

start_time = time.time()

# Serve static dashboard

app.mount("/web", StaticFiles(directory="web"), name="web")

# Load model on startup

model_path = "models/cifar10_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = create_cifar10_model()
    model = compile_model(model)
    save_model(model, model_path)

# Class names

class_names = [
'airplane', 'automobile', 'bird', 'cat', 'deer',
'dog', 'frog', 'horse', 'ship', 'truck'
]

@app.get("/", response_class=HTMLResponse)
def read_dashboard():
with open("web/dashboard.html", "r", encoding="utf-8") as f:
return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
img = Image.open(file.file).resize((32, 32)).convert("RGB")
x = np.array(img)
x = x / 255.0
x = np.expand_dims(x, axis=0)
preds = model.predict(x)
class_index = int(np.argmax(preds))
return {"class_index": class_index, "class_name": class_names[class_index]}

@app.post("/retrain")
async def retrain(files: list[UploadFile] = File(...)):
new_images = []
new_labels = []
for file in files:
filename = file.filename
class_name = filename.split("/")[0]  # expects folder/class_name/image.jpg
if class_name in class_names:
class_idx = class_names.index(class_name)
img = Image.open(file.file).resize((32, 32)).convert("RGB")
new_images.append(np.array(img))
new_labels.append(class_idx)
if not new_images:
return JSONResponse({"error": "No valid images found"}, status_code=400)

```
# Combine with a subset of CIFAR-10 train
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
sample_size = min(10000, len(X_train))
indices = np.random.choice(len(X_train), sample_size, replace=False)
X_combined = np.concatenate([X_train[indices], np.array(new_images)])
y_combined = np.concatenate([y_train[indices], np.array(new_labels).reshape(-1,1)])

X_processed, y_processed = preprocess_data(X_combined, y_combined)
X_test_processed, y_test_processed = preprocess_data(X_test, y_test)

model.fit(X_processed, y_processed, batch_size=128, epochs=5, validation_split=0.2, verbose=1)
save_model(model, model_path)

test_loss, test_acc = model.evaluate(X_test_processed, y_test_processed, verbose=0)
return {"test_accuracy": float(test_acc), "test_loss": float(test_loss)}
```

@app.get("/metrics")
def get_metrics():
if os.path.exists("logs/training_logs.json"):
with open("logs/training_logs.json", "r") as f:
metrics = json.load(f)
return metrics
return {"error": "No metrics found"}

@app.get("/uptime")
def uptime():
elapsed = int(time.time() - start_time)
return {"uptime": f"{elapsed // 3600}h {(elapsed % 3600) // 60}m {elapsed % 60}s"}

if **name** == "**main**":
uvicorn.run(app, host="0.0.0.0", port=8000)

