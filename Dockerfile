# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y build-essential libglib2.0-0 libsm6 libxrender1 libxext6 wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# create folders
RUN mkdir -p data/train data/sample logs models

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
