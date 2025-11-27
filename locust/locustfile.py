from locust import HttpUser, task, between

class MLUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        # ensure you have data/sample/sample1.png
        with open("data/sample/sample1.png", "rb") as f:
            files = {"file": ("sample1.png", f, "image/png")}
            self.client.post("/predict/", files=files)

