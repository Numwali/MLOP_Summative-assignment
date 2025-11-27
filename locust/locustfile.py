from locust import HttpUser, task, between

class MLApiUser(HttpUser):
    wait_time = between(1,3)

    @task
    def predict_image(self):
        with open("data/sample/sample1.png", "rb") as f:
            files = {"file": f}
            self.client.post("/predict", files=files)

