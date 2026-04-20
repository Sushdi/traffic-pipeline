"""
FastAPI app — exposes /predict, /health, and /metrics (Prometheus) endpoints.

Prometheus metrics exported:
  - traffic_predictions_total        (counter, by class)
  - traffic_prediction_latency_seconds (histogram)
  - traffic_prediction_errors_total  (counter)
  - traffic_live_accuracy            (gauge) ← from evaluate_live.py output
  - traffic_live_f1                  (gauge)
  - traffic_data_drift_psi           (gauge) ← PSI on current_speed
  - traffic_model_drift_score        (gauge) ← 1 - avg_max_confidence
  - traffic_eval_total               (gauge) ← total prediction–truth pairs
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response

log = logging.getLogger("uvicorn.error")

with open("params.yaml") as f:
    CFG = yaml.safe_load(f)

MODEL_PATH   = CFG["paths"]["model"]
FEATURE_COLS = None
MODEL_BUNDLE = None

# ── Prometheus metrics ────────────────────────────────────────────────────────
PREDICTION_COUNTER = Counter(
    "traffic_predictions_total",
    "Total predictions by class",
    ["congestion_level"],
)
PREDICTION_LATENCY = Histogram(
    "traffic_prediction_latency_seconds",
    "Prediction latency",
)
REQUEST_ERRORS = Counter(
    "traffic_prediction_errors_total",
    "Prediction errors",
)

# ── Live monitoring gauges (populated by /metrics/drift on each Prometheus scrape)
LIVE_ACCURACY = Gauge(
    "traffic_live_accuracy",
    "Latest 15-min-ahead prediction accuracy (0–1)",
)
LIVE_F1 = Gauge(
    "traffic_live_f1",
    "Latest weighted F1 score for 15-min-ahead predictions",
)
DATA_DRIFT_PSI = Gauge(
    "traffic_data_drift_psi",
    "Population Stability Index for current_speed vs historical reference",
)
MODEL_DRIFT_SCORE = Gauge(
    "traffic_model_drift_score",
    "Model confidence drift score: 1 − avg max-class probability",
)
EVAL_TOTAL = Gauge(
    "traffic_eval_total",
    "Total number of 15-min prediction–ground-truth comparison pairs logged",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_BUNDLE, FEATURE_COLS
    log.info("Loading model from %s ...", MODEL_PATH)
    MODEL_BUNDLE = joblib.load(MODEL_PATH)
    FEATURE_COLS = MODEL_BUNDLE["features"]
    log.info(
        "Model loaded: %s  (F1=%.4f)  horizon=%d min",
        MODEL_BUNDLE["best_model"],
        MODEL_BUNDLE["best_f1"],
        MODEL_BUNDLE.get("horizon_minutes", 15),
    )
    yield


app = FastAPI(
    title="Traffic Congestion Predictor — Munich",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / response schemas ────────────────────────────────────────────────

class TrafficFeatures(BaseModel):
    # Cyclical time
    hour_sin:  float = Field(..., ge=-1, le=1)
    hour_cos:  float = Field(..., ge=-1, le=1)
    dow_sin:   float = Field(..., ge=-1, le=1)
    dow_cos:   float = Field(..., ge=-1, le=1)
    min_sin:   float = Field(..., ge=-1, le=1)
    min_cos:   float = Field(..., ge=-1, le=1)
    # Peak flags
    is_weekend:       int = Field(..., ge=0, le=1)
    is_peak_morning:  int = Field(..., ge=0, le=1)
    is_peak_evening:  int = Field(..., ge=0, le=1)
    # Flow features
    current_speed:         float = Field(..., ge=0)
    free_flow_speed:       float = Field(..., gt=0)
    current_travel_time:   float = Field(..., ge=0)
    free_flow_travel_time: float = Field(..., gt=0)
    confidence:            float = Field(..., ge=0, le=1)
    road_closure:          int   = Field(..., ge=0, le=1)
    frc_code:              int   = Field(..., ge=0, le=7)
    # Lag features
    speed_lag_1:  float
    speed_lag_3:  float
    speed_lag_6:  float
    speed_lag_12: float
    # Rolling features
    speed_roll_mean_6:  float
    speed_roll_mean_12: float
    speed_roll_std_6:   float
    tt_ratio_roll_6:    float
    speed_trend:        float

class PredictionResponse(BaseModel):
    congestion_level: str
    probabilities:    dict[str, float]
    model_used:       str
    latency_ms:       float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status":     "ok",
        "model":      MODEL_BUNDLE["best_model"],
        "best_f1":    MODEL_BUNDLE["best_f1"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: TrafficFeatures):
    t0 = time.perf_counter()
    try:
        X = np.array([[getattr(data, f) for f in FEATURE_COLS]])
        clf = MODEL_BUNDLE["model"]
        le  = MODEL_BUNDLE["label_encoder"]

        pred_idx   = clf.predict(X)[0]
        pred_proba = clf.predict_proba(X)[0]
        label      = le.inverse_transform([pred_idx])[0]

        PREDICTION_COUNTER.labels(congestion_level=label).inc()
        latency = (time.perf_counter() - t0) * 1000

        return PredictionResponse(
            congestion_level=label,
            probabilities={
                cls: round(float(p), 4)
                for cls, p in zip(le.classes_, pred_proba)
            },
            model_used=MODEL_BUNDLE["best_model"],
            latency_ms=round(latency, 2),
        )
    except Exception as exc:
        REQUEST_ERRORS.inc()
        log.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint — prediction counters + latency."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/drift")
def metrics_drift():
    """
    Prometheus scrape endpoint for live accuracy & drift gauges.
    Reads the latest row from monitoring/eval_log.csv (written by
    src/evaluate_live.py) and updates the Gauge values before responding.

    Add this path to prometheus.yml under a second scrape job:
      - job_name: traffic-drift
        static_configs:
          - targets: ["api:8000"]
        metrics_path: /metrics/drift
    """
    eval_log = CFG["paths"].get("eval_log", "monitoring/eval_log.csv")
    if os.path.exists(eval_log):
        try:
            df = pd.read_csv(eval_log)
            if not df.empty:
                last = df.iloc[-1]
                LIVE_ACCURACY.set(float(last["accuracy"]))
                LIVE_F1.set(float(last["f1_weighted"]))
                EVAL_TOTAL.set(float(len(df)))

                psi = last.get("speed_psi", float("nan"))
                if not (isinstance(psi, float) and psi != psi):  # not NaN
                    DATA_DRIFT_PSI.set(float(psi))

                drift = last.get("model_drift_score", float("nan"))
                if not (isinstance(drift, float) and drift != drift):
                    MODEL_DRIFT_SCORE.set(float(drift))
        except Exception as exc:
            log.warning("Could not read eval_log for /metrics/drift: %s", exc)

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model-info")
def model_info():
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "best_model":      MODEL_BUNDLE["best_model"],
        "best_f1":         MODEL_BUNDLE["best_f1"],
        "features":        FEATURE_COLS,
        "classes":         list(MODEL_BUNDLE["label_encoder"].classes_),
        "target":          MODEL_BUNDLE.get("target", "target_15m"),
        "horizon_minutes": MODEL_BUNDLE.get("horizon_minutes", 15),
    }