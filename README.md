# CIFAR-10 MLOps Project

## Overview
End-to-end CIFAR-10 image classification MLOps project:
- Notebook: data exploration, preprocessing, training, evaluation
- Model: CNN saved as `models/cifar10_model_latest.keras`
- API: FastAPI app at `app/app.py` for predict, upload, retrain, metrics, uptime
- UI: Streamlit app at `app_ui/streamlit_app.py`
- Retraining: upload dataset (ZIP) -> extract to `data/train/` -> `/retrain/` triggers fine-tune
- Load testing: Locust config in `locust/locustfile.py`
- Docker: `Dockerfile` + `docker-compose.yml` for containerized deployment

---
**Demo Video link**: https://www.youtube.com/watch?v=dTch9ntPY_U
**Deployment link**: https://mlop-summative-assignment-qtyd.onrender.com

## Quick start (local)

1. **Create & activate virtualenv**
```bash
python -m venv venv
# Windows
venv\\Scripts\\activate
# mac / linux
source venv/bin/activate
```

2.**Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run FastAPI**

```bash
uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
# then visit http://127.0.0.1:8000/docs
```

4. **Run Streamlit UI
```bash
streamlit run app_ui/streamlit_app.py
# UI available at http://localhost:8501
```
 5. **Test Predict**
 - Use Swagger UI or:

```bash
curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@notebook/test_images/sample1.png"
```
6. **Upload data for retrain**

- Create a ZIP with folder layout like:
mydata.zip
  └─ dog/
     └─ img1.jpg
  └─ cat/
     └─ img2.jpg

- Post to /upload-data (via Swagger or UI). Then call /retrain/

7. **Load test with Locust**
```bash
locust -f locust/locustfile.py --host=http://127.0.0.1:8000

# Open http://localhost:8089 to run the flood
```

8. **Files Structure**
models/
  cifar10_model_latest.keras
  classes.json

data/
  train/        # retrain uploads extracted here
  sample/       # sample test images

src/
  preprocessing.py
  model.py
  prediction.py
  retrain.py

app/
  app.py
  utils.py

app_ui/
  streamlit_app.py

locust/
  locustfile.py

Dockerfile
docker-compose.yml
requirements.txt
```
