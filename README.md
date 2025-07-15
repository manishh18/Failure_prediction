# 🛠️ Predictive Maintenance Dashboard

Anticipate industrial machine failures before they happen! This project uses machine learning to predict both the occurrence and type of machine failures, helping reduce downtime, improve safety, and optimize maintenance schedules.

---

##  Project Overview

- **Binary Classification:** Predict if a machine will fail.
- **Multiclass Classification:** Identify the cause of failure (e.g., Overstrain, Power, Tool Wear, Heat Dissipation, Random).
- **Interactive Dashboard:** Explore data, visualize metrics, and make predictions via a Streamlit web app.
- **REST API:** FastAPI backend for programmatic predictions.

---

##  Dataset
- **10,000 samples** with 14 features: sensor readings, operational settings, and failure indicators.
- **Failure Modes:**
  - Tool Wear Failure (TWF)
  - Heat Dissipation Failure (HDF)
  - Power Failure (PWF)
  - Overstrain Failure (OSF)
  - Random Failure (RNF)

---

##  Features
- **EDA:** Visualize distributions, outliers, and correlations.
- **Model Metrics:** Compare classifiers (Random Forest, SVC, etc.) for both tasks.
- **Prediction:** Input sensor data and get real-time failure predictions and root cause.
- **API:** `/predict` endpoint for programmatic access.

---

##  Setup & Installation

### 1. Clone the repository
```bash
git clone <repo-url>
cd PM
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
- **Streamlit Dashboard:**
  ```bash
  streamlit run Home.py
  ```
- **FastAPI Backend:**
  ```bash
  uvicorn main:app --reload
  ```

> By default, Streamlit runs on [localhost:8501](http://localhost:8501) and FastAPI on [localhost:8000](http://localhost:8000).

---

##  Docker Deployment

Build and run everything (Streamlit + FastAPI) in one container:

```bash
docker build -t predictive-maintenance .
docker run -p 8501:8501 -p 8000:8000 predictive-maintenance
```

---

## 🔗 API Usage

- **Endpoint:** `POST /predict`
- **Request JSON:**
  ```json
  {
    "air_temperature_K": 300.0,
    "process_temperature_K": 310.0,
    "rotational_speed_rpm": 1500,
    "torque_Nm": 40.0,
    "tool_wear_min": 100,
    "type": "L"  // "L", "M", or "H"
  }
  ```
- **Response:**
  ```json
  { "prediction": "No Failure" }
  // or
  { "prediction": "Failure Detected", "failure_type": "Overstrain Failure" }
  ```

---

##  Dashboard Pages
- **Home:** Project intro, dataset, and failure mode explanations.
- **EDA:** Data exploration, outlier detection, and feature analysis.
- **Metrics:** Model performance comparison and classification reports.
- **Prediction:** Input form for real-time predictions.

---

##  Configuration
- Streamlit settings: see `.streamlit/config.toml` (runs headless on port 8501).
- No authentication or API keys required for local use.

---

##  Notebooks
- See `notebooks/` for data preprocessing, EDA, and model training workflows.

---

##  License
MIT License (add your own if needed)
