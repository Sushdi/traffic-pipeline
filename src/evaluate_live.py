"""
Live 15-minute-ahead evaluation loop.

How it works
------------
Every time this script runs (called every 5 min by a cron job or scheduler):

  1. Fetch current TomTom data for all waypoints  →  this is the GROUND TRUTH
     for predictions made 15 minutes ago.
  2. Look up the prediction stored 15 minutes ago in predictions_queue.db.
  3. Compare prediction vs ground truth  →  log accuracy, F1, drift metrics
     to monitoring/eval_log.csv.
  4. Build features from the CURRENT data.
  5. Make a NEW prediction for 15 min from now and store it in the queue.

Run via cron (every 5 minutes while data collection is active):
    */5 * * * * cd /path/to/traffic-pipeline && \
        TOMTOM_API_KEY=<key> .venv/bin/python src/evaluate_live.py >> logs/eval.log 2>&1
"""

import os
import math
import sqlite3
import logging
import time
from datetime import datetime, timezone, timedelta

import joblib
import numpy as np
import pandas as pd
import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
with open("params.yaml") as f:
    CFG = yaml.safe_load(f)

API_KEY        = os.environ["TOMTOM_API_KEY"]
MODEL_PATH     = CFG["paths"]["model"]
DB_PATH        = CFG["paths"]["predictions_db"]
EVAL_LOG_PATH  = CFG["paths"]["eval_log"]
WAYPOINTS      = CFG["waypoints"]
TIMEOUT        = CFG["collection"]["api_timeout"]
HORIZON_STEPS  = CFG["collection"]["prediction_horizon_steps"]   # 3
INTERVAL_SECS  = CFG["collection"]["interval_seconds"]            # 300
HORIZON_MINS   = (HORIZON_STEPS * INTERVAL_SECS) // 60           # 15

FREE_FLOW_THRESH = CFG["labeling"]["free_flow_threshold"]
MODERATE_THRESH  = CFG["labeling"]["moderate_threshold"]

# ── Load model bundle ─────────────────────────────────────────────────────────
bundle = joblib.load(MODEL_PATH)
MODEL  = bundle["model"]
LE     = bundle["label_encoder"]
FEATS  = bundle["features"]


# ── SQLite queue helpers ───────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions_queue (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT NOT NULL,        -- ISO UTC when prediction was made
            target_time     TEXT NOT NULL,        -- ISO UTC when it should be evaluated
            location_name   TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            prob_free_flow  REAL,
            prob_moderate   REAL,
            prob_congested  REAL,
            evaluated       INTEGER DEFAULT 0     -- 0=pending, 1=done
        )
    """)
    conn.commit()
    return conn


def save_predictions(conn: sqlite3.Connection, predictions: list[dict], target_time: datetime):
    """Store a batch of predictions with the time they are valid for."""
    target_iso = target_time.isoformat()
    created_iso = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            created_iso,
            target_iso,
            p["location_name"],
            p["predicted_label"],
            p.get("prob_free_flow"),
            p.get("prob_moderate"),
            p.get("prob_congested"),
        )
        for p in predictions
    ]
    conn.executemany(
        """INSERT INTO predictions_queue
           (created_at, target_time, location_name, predicted_label,
            prob_free_flow, prob_moderate, prob_congested)
           VALUES (?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    log.info("Saved %d predictions → target_time=%s", len(rows), target_iso)


def fetch_pending_predictions(conn: sqlite3.Connection, target_time: datetime) -> pd.DataFrame:
    """Return predictions whose target_time is within ±3 min of target_time."""
    window_lo = (target_time - timedelta(minutes=3)).isoformat()
    window_hi = (target_time + timedelta(minutes=3)).isoformat()
    rows = conn.execute(
        """SELECT id, location_name, predicted_label,
                  prob_free_flow, prob_moderate, prob_congested
           FROM predictions_queue
           WHERE target_time BETWEEN ? AND ?
             AND evaluated = 0""",
        (window_lo, window_hi),
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "id", "location_name", "predicted_label",
        "prob_free_flow", "prob_moderate", "prob_congested",
    ])
    return df


def mark_evaluated(conn: sqlite3.Connection, ids: list[int]):
    conn.executemany(
        "UPDATE predictions_queue SET evaluated=1 WHERE id=?",
        [(i,) for i in ids],
    )
    conn.commit()


# ── TomTom helpers ─────────────────────────────────────────────────────────────

def fetch_tomtom(lat: float, lon: float) -> dict | None:
    url = (
        "https://api.tomtom.com/traffic/services/4"
        f"/flowSegmentData/absolute/10/json"
        f"?point={lat},{lon}&key={API_KEY}"
    )
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()["flowSegmentData"]
    except requests.RequestException as exc:
        log.error("TomTom fetch failed (%s, %s): %s", lat, lon, exc)
        return None


def label_from_fsd(fsd: dict) -> str:
    cur_speed  = float(fsd["currentSpeed"])
    free_speed = float(fsd["freeFlowSpeed"])
    ratio = cur_speed / max(free_speed, 1.0)
    if ratio >= FREE_FLOW_THRESH:
        return "free_flow"
    elif ratio >= MODERATE_THRESH:
        return "moderate"
    return "congested"


def build_features(fsd: dict) -> dict:
    """Build the same feature vector used during training (no lag history in
    live mode — lags approximate to current speed; same logic as dashboard.py)."""
    now    = datetime.now(timezone.utc)
    hour   = now.hour
    minute = now.minute
    dow    = now.weekday()

    cur_speed  = float(fsd["currentSpeed"])
    free_speed = float(fsd["freeFlowSpeed"])
    cur_tt     = float(fsd["currentTravelTime"])
    free_tt    = float(fsd["freeFlowTravelTime"])

    frc_map  = {"FRC0": 0, "FRC1": 1, "FRC2": 2, "FRC3": 3,
                "FRC4": 4, "FRC5": 5, "FRC6": 6, "FRC7": 7}
    frc_code = frc_map.get(fsd.get("frc", ""), 4)

    return {
        "hour_sin":  math.sin(2 * math.pi * hour / 24),
        "hour_cos":  math.cos(2 * math.pi * hour / 24),
        "dow_sin":   math.sin(2 * math.pi * dow / 7),
        "dow_cos":   math.cos(2 * math.pi * dow / 7),
        "min_sin":   math.sin(2 * math.pi * minute / 60),
        "min_cos":   math.cos(2 * math.pi * minute / 60),
        "is_weekend":      int(dow >= 5),
        "is_peak_morning": int(7 <= hour <= 9),
        "is_peak_evening": int(16 <= hour <= 19),
        "current_speed":         cur_speed,
        "free_flow_speed":       free_speed,
        "current_travel_time":   cur_tt,
        "free_flow_travel_time": free_tt,
        "confidence":   float(fsd.get("confidence", 0.9)),
        "road_closure": int(bool(fsd.get("roadClosure", False))),
        "frc_code":     frc_code,
        # Lags approximated to current speed (no history buffer here)
        "speed_lag_1":  cur_speed,
        "speed_lag_3":  cur_speed,
        "speed_lag_6":  cur_speed,
        "speed_lag_12": cur_speed,
        "speed_roll_mean_6":  cur_speed,
        "speed_roll_mean_12": cur_speed,
        "speed_roll_std_6":   0.0,
        "tt_ratio_roll_6":    cur_tt / max(free_tt, 1),
        "speed_trend":        0.0,
    }


# ── Drift helpers ──────────────────────────────────────────────────────────────

def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for a numeric feature.
    PSI < 0.1  → no significant change
    PSI 0.1–0.2 → moderate shift
    PSI > 0.2  → significant shift (data drift)
    """
    try:
        breakpoints = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()),
            bins + 1,
        )
        expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_pct   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)
        # Avoid division by zero
        expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
        actual_pct   = np.where(actual_pct   == 0, 1e-4, actual_pct)
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(round(psi, 6))
    except Exception:
        return float("nan")


def load_reference_speeds() -> pd.Series | None:
    """Load historical current_speed values to use as PSI reference."""
    raw_path = CFG["paths"]["raw"]
    if not os.path.exists(raw_path):
        return None
    df = pd.read_csv(raw_path, usecols=["current_speed"])
    return df["current_speed"].dropna()


# ── Evaluation + logging ───────────────────────────────────────────────────────

def log_eval_row(eval_time: datetime, accuracy: float, f1: float,
                 n_correct: int, n_total: int,
                 label_dist_actual: dict, label_dist_predicted: dict,
                 speed_psi: float, model_drift_score: float,
                 per_location: list[dict]):
    """Append one evaluation result row to the CSV log."""
    os.makedirs(os.path.dirname(EVAL_LOG_PATH), exist_ok=True)
    row = {
        "eval_time":            eval_time.isoformat(),
        "accuracy":             round(accuracy, 4),
        "f1_weighted":          round(f1, 4),
        "n_correct":            n_correct,
        "n_total":              n_total,
        # Label distribution counts
        "actual_free_flow":     label_dist_actual.get("free_flow", 0),
        "actual_moderate":      label_dist_actual.get("moderate", 0),
        "actual_congested":     label_dist_actual.get("congested", 0),
        "pred_free_flow":       label_dist_predicted.get("free_flow", 0),
        "pred_moderate":        label_dist_predicted.get("moderate", 0),
        "pred_congested":       label_dist_predicted.get("congested", 0),
        # Drift signals
        "speed_psi":            speed_psi,          # data drift (feature)
        "model_drift_score":    model_drift_score,  # avg max-prob confidence drop
    }
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(EVAL_LOG_PATH)
    df_row.to_csv(EVAL_LOG_PATH, mode="a", header=write_header, index=False)
    log.info(
        "Logged eval at %s  accuracy=%.3f  f1=%.3f  speed_psi=%.4f  "
        "model_drift=%.4f  (%d/%d correct)",
        eval_time.strftime("%H:%M"), accuracy, f1, speed_psi, model_drift_score,
        n_correct, n_total,
    )

    # Also log per-location rows to a separate file for fine-grained analysis
    per_loc_path = EVAL_LOG_PATH.replace(".csv", "_per_location.csv")
    df_loc = pd.DataFrame(per_location)
    df_loc["eval_time"] = eval_time.isoformat()
    write_header_loc = not os.path.exists(per_loc_path)
    df_loc.to_csv(per_loc_path, mode="a", header=write_header_loc, index=False)


# ── Main cycle ────────────────────────────────────────────────────────────────

def run_cycle():
    """
    One full evaluation + prediction cycle:
      - Fetch current ground truth from TomTom
      - Evaluate the prediction made HORIZON_MINS minutes ago
      - Store a new prediction for HORIZON_MINS minutes from now
    """
    now  = datetime.now(timezone.utc)
    conn = get_conn()

    # ── Step 1: Fetch current TomTom data ────────────────────────────────────
    current_records = []
    for wp in WAYPOINTS:
        fsd = fetch_tomtom(wp["lat"], wp["lon"])
        if fsd is None:
            continue
        current_records.append({
            "location_name":  wp["name"],
            "actual_label":   label_from_fsd(fsd),
            "current_speed":  float(fsd["currentSpeed"]),
            "fsd":            fsd,
        })
        time.sleep(0.4)   # polite rate limiting

    if not current_records:
        log.warning("No TomTom data collected — skipping cycle.")
        conn.close()
        return

    df_current = pd.DataFrame(current_records)

    # ── Step 2: Retrieve old predictions that should be valid NOW ────────────
    # We look for predictions whose target_time ≈ now (within ±3 min)
    df_old_preds = fetch_pending_predictions(conn, now)

    if not df_old_preds.empty:
        merged = df_current.merge(df_old_preds, on="location_name", how="inner")

        if not merged.empty:
            n_total   = len(merged)
            n_correct = (merged["actual_label"] == merged["predicted_label"]).sum()
            accuracy  = n_correct / n_total

            # F1 (weighted)
            try:
                from sklearn.metrics import f1_score
                f1 = f1_score(
                    merged["actual_label"],
                    merged["predicted_label"],
                    average="weighted",
                    zero_division=0,
                )
            except Exception:
                f1 = float("nan")

            label_dist_actual    = merged["actual_label"].value_counts().to_dict()
            label_dist_predicted = merged["predicted_label"].value_counts().to_dict()

            # ── Data drift: PSI on current_speed vs reference ────────────────
            ref_speeds = load_reference_speeds()
            speed_psi  = (
                compute_psi(ref_speeds, df_current["current_speed"])
                if ref_speeds is not None and len(df_current) > 1
                else float("nan")
            )

            # ── Model drift: drop in prediction confidence ────────────────────
            # If the model is drifting, its max probability starts falling
            # towards the uniform baseline (1/3 ≈ 0.333 for 3 classes).
            prob_cols = ["prob_free_flow", "prob_moderate", "prob_congested"]
            if all(c in merged.columns for c in prob_cols):
                avg_max_prob      = merged[prob_cols].max(axis=1).mean()
                # Ideal confidence ≈ 1.0; uniform random ≈ 0.333
                model_drift_score = round(1.0 - avg_max_prob, 4)
            else:
                model_drift_score = float("nan")

            # ── Per-location detail rows ──────────────────────────────────────
            per_location = merged[[
                "location_name", "actual_label", "predicted_label",
                "current_speed",
            ]].copy()
            per_location["correct"] = (
                per_location["actual_label"] == per_location["predicted_label"]
            ).astype(int)
            per_location_list = per_location.to_dict("records")

            log_eval_row(
                eval_time=now,
                accuracy=accuracy,
                f1=f1,
                n_correct=int(n_correct),
                n_total=n_total,
                label_dist_actual=label_dist_actual,
                label_dist_predicted=label_dist_predicted,
                speed_psi=speed_psi,
                model_drift_score=model_drift_score,
                per_location=per_location_list,
            )

            # Mark these predictions as evaluated in the queue
            mark_evaluated(conn, df_old_preds["id"].tolist())
        else:
            log.info("Predictions found but no location overlap — skipping eval.")
    else:
        log.info(
            "No predictions found for target_time ≈ %s. "
            "(Normal for first %d minutes of operation.)",
            now.strftime("%H:%M"), HORIZON_MINS,
        )

    # ── Step 3: Make NEW predictions for HORIZON_MINS minutes from now ───────
    target_time = now + timedelta(minutes=HORIZON_MINS)
    new_predictions = []

    for rec in current_records:
        fsd  = rec["fsd"]
        feat = build_features(fsd)
        X    = np.array([[feat[f] for f in FEATS]])

        pred  = MODEL.predict(X)[0]
        proba = MODEL.predict_proba(X)[0]
        label = LE.inverse_transform([pred])[0]

        classes = list(LE.classes_)
        prob_map = dict(zip(classes, proba.tolist()))

        new_predictions.append({
            "location_name":  rec["location_name"],
            "predicted_label": label,
            "prob_free_flow":  prob_map.get("free_flow", 0.0),
            "prob_moderate":   prob_map.get("moderate",  0.0),
            "prob_congested":  prob_map.get("congested", 0.0),
        })
        log.info(
            "  %-22s  predicted=%s (in 15 min)  confidence=%.1f%%",
            rec["location_name"], label,
            max(proba) * 100,
        )

    save_predictions(conn, new_predictions, target_time)
    conn.close()
    log.info("Cycle complete. Next evaluation at ~%s.", target_time.strftime("%H:%M"))


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  evaluate_live.py  —  15-min prediction + evaluation")
    log.info("  Horizon: %d min  |  %d waypoints", HORIZON_MINS, len(WAYPOINTS))
    log.info("=" * 60)
    run_cycle()
