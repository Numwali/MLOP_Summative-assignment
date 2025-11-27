import streamlit as st
import requests
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import base64

API_URL = "http://127.0.0.1:8000"

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="CIFAR-10 ML System",
    layout="wide",
    page_icon="🔵"
)

st.markdown("""
    <h1 style='text-align: center; color: #2B6CB0;'>CIFAR-10 Image Classification System</h1>
    <h4 style='text-align: center; color: gray;'>Prediction • Retraining • Monitoring</h4>
""", unsafe_allow_html=True)

# ---------- SIDEBAR NAV ----------
page = st.sidebar.radio("Navigation", ["Prediction", "Retrain Model", "Monitoring Dashboard"])

# ======================================================================
# PAGE 1 — PREDICTION
# ======================================================================
if page == "Prediction":
    st.subheader("🔵 Image Prediction")

    uploaded = st.file_uploader("Upload an image (CIFAR-10)", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", width=200)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        if st.button("Predict"):
            with st.spinner("Sending to model..."):
                files = {"file": ("image.png", buf, "image/png")}
                response = requests.post(f"{API_URL}/predict/", files=files)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: **{result['class']}**")
                st.info(f"Confidence: {result['confidence']:.2f}%")
            else:
                st.error("Error during prediction.")


# ======================================================================
# PAGE 2 — RETRAINING
# ======================================================================
elif page == "Retrain Model":
    st.subheader(" Retrain Model")

    st.info("Upload a ZIP file containing class folders (e.g., airplane/, car/, dog/)")

    zip_file = st.file_uploader("Upload ZIP dataset", type=["zip"])

    if zip_file:
        if st.button("Upload Dataset"):
            files = {"file": ("data.zip", zip_file, "application/zip")}
            response = requests.post(f"{API_URL}/upload-data/", files=files)

            if response.status_code == 200:
                st.success("Dataset uploaded successfully!")

    st.markdown("---")
    st.subheader("Trigger Retraining")

    if st.button("Start Retraining Now"):
        with st.spinner("Retraining model... This may take several minutes"):
            response = requests.post(f"{API_URL}/retrain/")

        if response.status_code == 200:
            metrics = response.json()
            st.success("Retraining completed!")
            st.write(metrics)

            # Plot accuracy
            fig, ax = plt.subplots()
            ax.plot(metrics["accuracy_curve"])
            ax.set_title("Accuracy Curve")
            st.pyplot(fig)

            # Plot loss
            fig, ax = plt.subplots()
            ax.plot(metrics["loss_curve"])
            ax.set_title("Loss Curve")
            st.pyplot(fig)


# ======================================================================
# PAGE 3 — MONITORING DASHBOARD
# ======================================================================
elif page == "Monitoring Dashboard":
    st.subheader(" Model Monitoring & Insights")

    # Uptime
    uptime = requests.get(f"{API_URL}/uptime/").json()
    st.metric("API Uptime (seconds)", value=int(uptime["uptime"]))

    # Metrics
    metrics = requests.get(f"{API_URL}/metrics/").json()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        st.metric("Loss", f"{metrics['loss']:.4f}")

    with col2:
        st.write("Class Distribution")
        st.bar_chart(metrics["class_distribution"])

    st.markdown("---")
    st.subheader("Feature Interpretations (Explainability)")

    st.write("### Grad-CAM Heatmap Example (Model Focus Area)")
    st.image(metrics["gradcam"], width=350)

    st.write("### Filters Learned by CNN (Early Layers)")
    st.image(metrics["filters"], width=350)

    st.write("### Confusion Samples (Where Model Struggles)")
    st.image(metrics["confused"], width=350)

