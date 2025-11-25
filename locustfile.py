"""
locustfile.py

Locust load test that posts a sample image to the /predict endpoint.
To run locally:
1. Ensure the server is running (http://localhost:8000)
2. `locust -f locustfile.py --host=http://127.0.0.1:8000`
"""

from locust import HttpUser, task, between
import random
import os

SAMPLE_DIR = "web"  # place a sample image in web/sample.jpg or use any PNG/JPEG
SAMPLE_FILE = "sample_image.jpg"

class CIFARUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(1)
    def predict_image(self):
        # choose a sample image from web/ or notebook/logs
        path = SAMPLE_FILE
        if not os.path.exists(path):
            # try fallback to notebook/logs/sample_images.png
            path = "notebook/logs/sample_images.png"
        if not os.path.exists(path):
            return

        with open(path, "rb") as f:
            files = {"image": ("sample.jpg", f, "image/jpeg")}
            self.client.post("/predict", files=files, timeout=30)

