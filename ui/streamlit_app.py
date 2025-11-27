import streamlit as st
import numpy as np
from PIL import Image
from src.prediction import load_latest_model, predict_image
from src.retrain import retrain_from_directory
import os

st.set_page_config(page_title="CIFAR-10 Classifier", layout="wide")
st.title("CIFAR-10 Image Classification Dashboard")

# --- Section 1: Upload and predict ---
st.header("Predict an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    model = load_latest_model()
    result = predict_image(model, np.array(img.resize((32,32))))
    
    st.subheader("Prediction Result")
    st.write(result)

# --- Section 2: Retrain the model ---
st.header("Retrain Model with New Data")
retrain_dir = "data/retrain"
if st.button("Trigger Retraining"):
    if not os.path.exists(retrain_dir):
        st.warning(f"No retrain data found in {retrain_dir}")
    else:
        retrain_log = retrain_from_directory(retrain_dir)
        st.success("Retraining completed!")
        st.json(retrain_log)

# --- Section 3: Model uptime / health check ---
st.header("Model Health Check")
if st.button("Check Uptime"):
    try:
        model = load_latest_model()
        st.success("Model is loaded and ready!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

