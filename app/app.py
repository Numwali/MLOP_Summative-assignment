from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from src.prediction import load_latest_model, predict_image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = FastAPI(title="CIFAR-10 Image Classification API")

# Load model once at startup
model = load_latest_model()

@app.get("/")
def root():
    return {"message": "CIFAR-10 Image Classification API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load image from uploaded file
        img = load_img(file.file, target_size=(32,32))
        img_array = img_to_array(img)
        result = predict_image(model, img_array)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

