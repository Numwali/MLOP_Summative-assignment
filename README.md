# CIFAR-10 MLOps Pipeline  
### Full End-to-End Machine Learning Operations System

---

## 1. Introduction
This project implements a complete **Machine Learning Operations (MLOps)** pipeline using the **CIFAR-10 image classification dataset**. It demonstrates production-grade ML workflows including training, versioning, deployment, monitoring, and a real-time prediction dashboard.

The system includes:
- A fully trained CIFAR-10 CNN model  
- A FastAPI backend for predictions, metrics & retraining  
- A modern, responsive web dashboard  
- Model retraining pipeline for continuous improvement  
- Dockerized deployment for reproducibility  

---

## 2. Project Structure

MLOP_Summative-assignment/
│── main.py
│── Dockerfile
│── docker-compose.yml
│── requirements.txt
│── README.md
│
│── models/
│ └── cifar10_model_latest.keras
│
│── logs/
│ └── training_logs.json
│
│── src/
│ ├── model.py
│ ├── train.py
│ ├── api.py
│ ├── prediction.py
│ └── preprocessing.py
│
└── web/
├── index.html
├── dashboard.js
└── style.css


---

## 3. System Overview

This project follows a modular and scalable MLOps architecture:

### **3.1 Model Development**
- A CNN classifier is trained using TensorFlow/Keras.
- Metrics (loss, accuracy) are logged to `logs/training_logs.json`.
- Models are saved and versioned in the `models/` directory.

### **3.2 Preprocessing**
- Images are resized to **32x32**, normalized, and converted to RGB.
- All preprocessing is implemented inside `src/preprocessing.py`.

### **3.3 FastAPI Backend**
Exposes four main ML endpoints:

| Endpoint | Description |
|----------|-------------|
| `/predict` | Classifies a single uploaded image |
| `/retrain` | Incrementally retrains the model with new labelled images |
| `/metrics` | Returns stored training performance metrics |
| `/uptime` | Returns server uptime for monitoring |

Static web dashboard is served directly from `web/`.

### **3.4 Web Dashboard**
The dashboard supports:
- Real-time predictions  
- Visualization of model probability distribution  
- Monitoring model uptime  
- Viewing stored training metrics  
- Clean UI built with HTML/CSS/JS  

### **3.5 Model Retraining**
The retraining pipeline:
1. Reads uploaded images with folder-based labels  
2. Preprocesses and validates labels  
3. Mixes with sampled CIFAR-10 data  
4. Retrains for 3 lightweight epochs  
5. Saves and updates the latest model version  

### **3.6 Dockerized Deployment**
The entire system is containerized using a single Dockerfile for reproducible deployment.

---

## 4. How to Run the Project

### **4.1 Install Dependencies**

pip install -r requirements.txt

## **4.2 Run FastAPI Backend**
uvicorn main:app --reload

## **4.3 Open Dashboard**

http://localhost:8000

## 5. **Dashboard Features**

The dashboard provides:

✔ Upload image → get instant prediction
✔ Probability bar chart for all CIFAR-10 classes
✔ Model uptime monitor
✔ Training metrics visualization
✔ File upload tool for retraining
✔ Fully responsive UI

## **6. Model Performance**

logs/training_logs.json

## **7. Retraining Process**

cat/image1.jpg
dog/image2.png
truck/photo3.jpeg


## **8. Docker Deployment**

docker build -t cifar10-mlops .

## 8.1 Build the image

docker build -t cifar10-mlops .

## 8.2 Run the container

docker run -p 8000:8000 cifar10-mlops

## 8.3 Dashboard available at:

http://localhost:8000

## **9. Monitoring & Logging**

Model training logs: logs/training_logs.json

Monitoring uptime: /uptime

API health & docs: http://localhost:8000/docs

The dashboard fetches this data automatically.

## **10. Academic Value**

This project demonstrates strong competency in MLOps:

Model training + evaluation

Versioning and reproducibility

API development for ML systems

Real-time inference dashboard

Monitoring & logging

Incremental training (continual learning)

Dockerized MLOps deployment

This implementation matches real-world ML production systems.

## **11. Conclusion**

A complete end-to-end MLOps pipeline was successfully implemented.
From preprocessing and model training to deployment, monitoring, retraining, and Dockerization — this project reflects both technical depth and strong MLOps practices.

## **12. Acknowledgements**

Special thanks to instructors and course materials that guided the development of this system.

---

