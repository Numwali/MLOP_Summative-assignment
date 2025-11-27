
---

## 2) Dockerfile  
**Location:** `Dockerfile` (root)

```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app code
COPY . .

# expose ports for API and streamlit
EXPOSE 8000 8501

# default to run uvicorn (API). For UI or others, override command in docker-compose
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]

