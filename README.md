
# CIFAR-10 Image Classification — End-to-End ML Pipeline

**Author:** Noella Umwali  
**Course:** Machine Learning / MLOps Summative Assignment

**Project summary:** This repository demonstrates a complete, reproducible pipeline for image classification using the CIFAR-10 dataset. It includes data preprocessing, model development (CNN), evaluation, retraining (triggerable through an API), an interactive web dashboard (frontend), and containerization for deployment. The goal is to present a professional project for modeling, retraining, monitoring, and deployment.

---

## Project Structure (recommended)

MLOP_Summative-assignment/
├── README.md
├── Dockerfile
├── requirements.txt
├── main.py # FastAPI app (serves API + static dashboard)
├── web/ # Frontend files (index.html, dashboard.js, CSS, images)
├── src/
│ ├── preprocessing.py
│ ├── model.py
│ ├── prediction.py # optional helper used by API
│ └── train.py # training/retraining scripts used in notebook
├── notebook/
│ └── CIFAR_10_Image_Classification.ipynb
├── models/ # saved models (.keras)
└── data/ # retrain data (class subfolders for upload)


---

## Quick start — Local (development) environment

1. **Clone the repo** (if not already):
   ```bash
   git clone <your-repo-url>
   cd MLOP_Summative-assignment

2. **Create and activate a virtual environment**
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows (Git Bash or PowerShell)

3. **Install dependencies**
pip install --upgrade pip
pip install -r requirements.txt

4. **Run the API (development)**
uvicorn main:app --reload

5. **To run in Docker**

docker build -t cifar-api:latest .
docker run --rm -p 8000:8000 cifar-api:latest


---
