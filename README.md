# Munich Traffic Congestion Predictor

#### A full MLOps pipeline that collects real-time traffic data from the TomTom API, trains a machine learning model to predict congestion levels, and serves predictions through a REST API and interactive dashboard.
---

## Pipeline overview

#### TomTom API → Data Collection → Feature Engineering → Model Training → FastAPI → Dashboard (GitHub Actions)    (preprocess.py)     (train.py)     (Docker)  (Streamlit)
---

## Project structure

traffic-pipeline/
├── .github/
│   └── workflows/
│       ├── collect_data.yml     # Runs every 5 min for 3 days, auto-chains
│       └── retrain.yml          # Nightly model retraining at 3AM UTC
├── data/
│   ├── raw/                     # Raw TomTom API data (appended every 5 min)
│   └── processed/               # Feature-engineered dataset
├── models/
│   └── model.pkl                # Trained model bundle (LightGBM)
├── metrics/
│   └── scores.json              # F1, AUC, log loss scores
├── monitoring/
│   └── prometheus.yml           # Prometheus scrape config
├── src/
│   ├── collect.py               # TomTom API collector (8 Munich waypoints)
│   ├── preprocess.py            # Feature engineering pipeline
│   ├── train.py                 # Trains RF / XGBoost / LightGBM + MLflow
│   ├── live_predict.py          # Sends live TomTom data to /predict
│   ├── make_fake_data.py        # Generates synthetic data for testing
│   └── api/
│       └── main.py              # FastAPI app
├── dashboard.py                 # Streamlit dashboard
├── params.yaml                  # Central config (waypoints, thresholds, paths)
├── Dockerfile                   # API container
├── docker-compose.yml           # API + Prometheus
└── requirements.txt

---

## Setup

### 1. Clone and create virtual environment
```bash
git clone https://github.com/Sushdi/traffic-pipeline.git
cd traffic-pipeline
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Set your TomTom API key
```bash
# Windows
$env:TOMTOM_API_KEY="your_key_here"

# Mac/Linux
export TOMTOM_API_KEY="your_key_here"
```

### 3. Run the full pipeline locally

```bash
# Collect one snapshot (8 Munich locations)
python src/collect.py

# Feature engineering
python src/preprocess.py

# Train models (RF, XGBoost, LightGBM)
python src/train.py

# View MLflow experiment results
mlflow ui
# → open http://127.0.0.1:5000
```

---

## Running the API

### Option A — Direct (uvicorn)
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option B — Docker
```bash
docker build -t traffic-api:latest .
docker run -p 8000:8000 traffic-api:latest
```

API endpoints:

GET  /health       → model status
POST /predict      → congestion prediction
GET  /metrics      → Prometheus metrics
GET  /model-info   → feature list + classes
GET  /docs         → interactive Swagger UI

Example prediction request:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hour_sin": -0.866, "hour_cos": -0.5,
    "dow_sin": 0.782,   "dow_cos": 0.623,
    "min_sin": 0.0,     "min_cos": 1.0,
    "is_weekend": 0, "is_peak_morning": 1, "is_peak_evening": 0,
    "current_speed": 18, "free_flow_speed": 50,
    "current_travel_time": 420, "free_flow_travel_time": 180,
    "confidence": 0.85, "road_closure": 0, "frc_code": 3,
    "speed_lag_1": 20, "speed_lag_3": 25,
    "speed_lag_6": 30, "speed_lag_12": 45,
    "speed_roll_mean_6": 24, "speed_roll_mean_12": 32,
    "speed_roll_std_6": 4.2, "tt_ratio_roll_6": 2.1,
    "speed_trend": -6.0
  }'
```

---

## Running the dashboard
```bash
# API must be running first (see above)
streamlit run dashboard.py
# → open http://localhost:8501
```

Dashboard tabs:
- **Live Traffic Map** — Folium heatmap + live predictions for all 8 locations
- **Model Performance** — Confusion matrix, ROC-AUC, Precision-Recall curves
- **Feature Analysis** — Feature importance, speed distributions, hourly patterns
- **Historical Trends** — Speed over time, day-of-week breakdown, weekend vs weekday
- **Now vs Usual** — Current traffic vs historical average for same hour + day type

---

## GitHub Actions (automated)

### Data collection
Triggered manually on Friday night — runs for 72 hours, collecting every 5 minutes.
Each 6-hour job chains automatically to the next.

Actions → Collect Traffic Data → Run workflow

Before triggering, update the `COLLECTION_START_EPOCH` variable:
```bash
python -c "import time; print(int(time.time()))"
# paste output into: GitHub → Settings → Variables → COLLECTION_START_EPOCH
```

### Nightly retraining
Runs automatically at 3AM UTC every day via `retrain.yml`.
Can also be triggered manually:

Actions → Retrain and Deploy → Run workflo

### Required GitHub secrets
TOMTOM_API_KEY   → TomTom API key
GH_PAT           → GitHub Personal Access Token (repo + workflow scopes)

---

## Model details

| Model | CV F1 | Test F1 | Log Loss | AUC |
|---|---|---|---|---|
| Random Forest | 0.9935 | 0.9940 | 0.0219 | 1.000 |
| XGBoost | 0.9966 | 0.9986 | 0.0045 | 1.000 |
| **LightGBM** | **0.9958** | **0.9993** | **0.0025** | **1.000** |

**Target classes:**
- `free_flow` — speed ratio ≥ 0.85 (88% of data)
- `moderate` — speed ratio 0.60–0.85 (11% of data)
- `congested` — speed ratio < 0.60 (1% of data)

**Features used:** cyclical time encoding, peak hour flags, raw TomTom flow metrics,
speed lag features (5/15/30/60 min), rolling averages and std dev.

**Data:** 7,184 rows collected over 3 days (Saturday–Monday) across 8 Munich locations
covering city centre, suburbs, Autobahn segments, and Mittlerer Ring.

---

## Monitoring

Prometheus scrapes `/metrics` every 30 seconds. Run with docker-compose:
```bash
docker-compose up
# Prometheus UI → http://localhost:9090
```

Tracked metrics:
- `traffic_predictions_total` — prediction count by congestion class
- `traffic_prediction_latency_seconds` — prediction latency histogram
- `traffic_prediction_errors_total` — error count

---

## Tech stack

| Layer | Tool |
|---|---|
| Data collection | TomTom Flow API + GitHub Actions |
| Data versioning | Git |
| Feature engineering | pandas, numpy |
| Model training | scikit-learn, XGBoost, LightGBM |
| Experiment tracking | MLflow |
| API | FastAPI + uvicorn |
| Containerisation | Docker |
| Dashboard | Streamlit + Plotly + Folium |
| Monitoring | Prometheus |
| CI/CD | GitHub Actions |
