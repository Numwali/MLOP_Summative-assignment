from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from src.prediction import Predictor
from src.retrain import retrain_from_directory
import shutil, os

app = FastAPI(title="CIFAR-10 ML API")

predictor = Predictor()

@app.get("/")
def root():
    return {"message": "CIFAR-10 Prediction API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    result = predictor.predict_from_bytes(img_bytes)
    return JSONResponse(result)

@app.post("/retrain")
def retrain():
    retrain_log = retrain_from_directory()
    if retrain_log is None:
        return JSONResponse({"status":"No new data for retraining."})
    return JSONResponse({"status":"Retraining completed", "log": retrain_log})

