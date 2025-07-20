# How I Deployed My FastAPI + Streamlit ML App on GCP (Step-by-Step)

Here's exactly how I got my machine learning project running on Google Cloud Platform using Cloud Run. I split the backend (FastAPI) and frontend (Streamlit) into two services for maximum reliability and scalability. If you want to do the same, just follow along!

---

## 1. My Project Structure

First, I organized my project like this:
```
/PM
  |-- main.py           # FastAPI app (backend)
  |-- Home.py           # Streamlit app (frontend)
  |-- requirements.txt  # Shared dependencies
  |-- Dockerfile.fastapi
  |-- Dockerfile.streamlit
  |-- ... (models, data, etc.)
```

---

## 2. Writing the Dockerfiles

### A. FastAPI Dockerfile (`Dockerfile.fastapi`)
This is how I containerized my FastAPI backend:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### B. Streamlit Dockerfile (`Dockerfile.streamlit`)
And here’s the Dockerfile for my Streamlit frontend:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["streamlit", "run", "Home.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

---

## 3. Building and Pushing Docker Images

I made sure my GCP project was set:
```sh
gcloud config set project predative-machine
```

Then I authenticated Docker with GCP:
```sh
gcloud auth configure-docker
```

**To build and push the FastAPI image:**
```sh
docker build -f Dockerfile.fastapi -t gcr.io/preditive-machine/pm-fastapi:latest .
docker push gcr.io/preditive-machine/pm-fastapi:latest
```

**To build and push the Streamlit image:**
```sh
docker build -f Dockerfile.streamlit -t gcr.io/preditive-machine/pm-streamlit:latest .
docker push gcr.io/preditive-machine/pm-streamlit:latest
```

---

## 4. Deploying to Cloud Run

**For FastAPI:**
```sh
gcloud run deploy pm-fastapi \
  --image gcr.io/preditive-machine/pm-fastapi:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080
```

**For Streamlit:**
```sh
gcloud run deploy pm-streamlit \
  --image gcr.io/predative-machine/pm-streamlit:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080
```

---

## 5. Connecting Streamlit to FastAPI

In my Streamlit code, I set the API URL to the FastAPI Cloud Run URL I got after deployment:
```python
url = "https://pm-fastapi-29369688553.us-central1.run.app/predict"
```
If I changed this, I always rebuilt and redeployed the Streamlit service.

---

## 6. Testing Everything

I visited my Streamlit Cloud Run URL in the browser and used the prediction form. It called the FastAPI backend and returned results. I also tested FastAPI directly using its `/docs` Swagger UI.

---

## 7. How I Monitor Costs and Usage

- I set up budgets and alerts in [GCP Budgets & Alerts](https://console.cloud.google.com/billing/budgets) so I never get surprised by charges.
- I check [GCP Billing Reports](https://console.cloud.google.com/billing/reports) to see where my money is going.
- I use the Cloud Run free tier as much as possible.
- I regularly clean up unused services and images to avoid extra costs.

---

**That’s how I got my ML app running on GCP! If you follow these steps, you’ll have a scalable, cloud-native deployment too.** 
