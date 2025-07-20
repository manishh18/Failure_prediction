# Predictive Maintenance ML App (GCP Deployed)

[![Live Demo](https://img.shields.io/badge/Live%20App-Streamlit-green)](https://pm-streamlit-29369688553.us-central1.run.app/)

## 🚀 Live Demo
**Try the app here:** [https://pm-streamlit-29369688553.us-central1.run.app/](https://pm-streamlit-29369688553.us-central1.run.app/)

---

## 📝 Project Overview
This project is an end-to-end machine learning solution for predictive maintenance. It predicts machine failures and identifies their root causes using real-world sensor data. The app is fully containerized and deployed on Google Cloud Platform (GCP) using Cloud Run for both the backend and frontend.

**Key Features:**
- **Interactive Streamlit dashboard** for data exploration, model evaluation, and live predictions
- **FastAPI backend** serving ML predictions via a REST API
- **Cloud-native deployment**: Both services run as independent, scalable Cloud Run services
- **Easy integration**: The frontend calls the backend via public HTTPS endpoints

---

## 🏗️ Architecture

- **Frontend:** Streamlit app ([Live URL](https://pm-streamlit-29369688553.us-central1.run.app/))
- **Backend:** FastAPI app (deployed on Cloud Run, private URL)
- **Communication:** Streamlit sends prediction requests to FastAPI via REST API
- **Deployment:** Both services are containerized with Docker and deployed to GCP Cloud Run

**Diagram:**
```
User ──▶ Streamlit (Cloud Run) ──▶ FastAPI (Cloud Run) ──▶ ML Model
```

---

## ⚙️ Setup & Deployment

**Quick Start:**
- Visit the [Live Streamlit App](https://pm-streamlit-29369688553.us-central1.run.app/)
- For local development or redeployment, see the full [GCP_DEPLOYMENT.md](./GCP Deployment.md)

**Main Steps:**
1. Write Dockerfiles for both FastAPI and Streamlit
2. Build and push images to Google Container Registry
3. Deploy each service to Cloud Run
4. Set the FastAPI endpoint URL in the Streamlit code

---

## 🔗 API Usage Example

**Endpoint:** `POST /predict` (FastAPI backend)

**Sample Request:**
```json
{
  "air_temperature_K": 300,
  "process_temperature_K": 310,
  "rotational_speed_rpm": 1500,
  "torque_Nm": 40,
  "tool_wear_min": 100,
  "type": "L"
}
```

**Sample Response:**
```json
{
  "prediction": "No Failure"
}
// or
{
  "prediction": "Failure Detected",
  "failure_type": "Overstrain Failure"
}
```

---

## 💸 Cost Monitoring & Best Practices
- Set up [GCP Budgets & Alerts](https://console.cloud.google.com/billing/budgets) to avoid overspending
- Monitor usage in [GCP Billing Reports](https://console.cloud.google.com/billing/reports)
- Use the Cloud Run free tier as much as possible
- Regularly clean up unused services and images

---

## 👤 Author
- **Manish**

---

**Enjoy exploring predictive maintenance with a modern, cloud-native ML stack!** 
