# locust/locustfile.py
from locust import HttpUser, task, between

class MLUser(HttpUser):
    wait_time = between(0.5, 2)

    @task
    def predict(self):
        with open("data/sample/sample1.png", "rb") as f:
            files = {"file": ("sample1.png", f, "image/png")}
            self.client.post("/predict/", files=files)


