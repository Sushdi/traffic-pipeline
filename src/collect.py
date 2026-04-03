"""
Traffic data collector — hits TomTom flowSegmentData for each Munich waypoint.
Run once per snapshot (called in a loop by the GitHub Actions job).
"""

import os
import time
import logging
from datetime import datetime, timezone

import requests
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
with open("params.yaml") as f:
    CFG = yaml.safe_load(f)

API_KEY   = os.environ["TOMTOM_API_KEY"]
RAW_PATH  = CFG["paths"]["raw"]
WAYPOINTS = CFG["waypoints"]
TIMEOUT   = CFG["collection"]["api_timeout"]

FREE_FLOW_THRESH = CFG["labeling"]["free_flow_threshold"]
MODERATE_THRESH  = CFG["labeling"]["moderate_threshold"]


# ── Core functions ───────────────────────────────────────────────────────────

def fetch(lat: float, lon: float) -> dict | None:
    url = (
        "https://api.tomtom.com/traffic/services/4"
        f"/flowSegmentData/absolute/10/json"
        f"?point={lat},{lon}&key={API_KEY}"
    )
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        log.error("fetch failed (%s, %s): %s", lat, lon, exc)
        return None


def parse(raw: dict, wp: dict) -> dict | None:
    try:
        fsd = raw["flowSegmentData"]
        now = datetime.now(timezone.utc)

        cur_speed  = float(fsd["currentSpeed"])
        free_speed = float(fsd["freeFlowSpeed"])
        cur_tt     = float(fsd["currentTravelTime"])
        free_tt    = float(fsd["freeFlowTravelTime"])

        speed_ratio = cur_speed / max(free_speed, 1.0)
        tt_ratio    = cur_tt   / max(free_tt,    1.0)

        # Label
        if speed_ratio >= FREE_FLOW_THRESH:
            label = "free_flow"
        elif speed_ratio >= MODERATE_THRESH:
            label = "moderate"
        else:
            label = "congested"

        return {
            # Identifiers
            "timestamp":             now.isoformat(),
            "location_name":         wp["name"],
            "latitude":              wp["lat"],
            "longitude":             wp["lon"],
            # Time features (raw — rolling/cyclical added in preprocess)
            "hour":                  now.hour,
            "minute":                now.minute,
            "day_of_week":           now.weekday(),   # 0=Mon … 6=Sun
            "is_weekend":            int(now.weekday() >= 5),
            # TomTom flow features
            "frc":                   fsd.get("frc", ""),
            "current_speed":         cur_speed,
            "free_flow_speed":       free_speed,
            "current_travel_time":   cur_tt,
            "free_flow_travel_time": free_tt,
            "confidence":            float(fsd.get("confidence", 0.0)),
            "road_closure":          int(bool(fsd.get("roadClosure", False))),
            # Derived
            "speed_ratio":           round(speed_ratio, 4),
            "travel_time_ratio":     round(tt_ratio,    4),
            # Target label
            "congestion_level":      label,
        }
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        log.error("parse failed for %s: %s", wp["name"], exc)
        return None


def collect_snapshot() -> int:
    """Collect one snapshot across all waypoints. Returns row count added."""
    records = []
    for wp in WAYPOINTS:
        raw = fetch(wp["lat"], wp["lon"])
        if raw:
            rec = parse(raw, wp)
            if rec:
                records.append(rec)
                log.info(
                    "  %-20s  speed_ratio=%.2f  label=%s",
                    wp["name"], rec["speed_ratio"], rec["congestion_level"],
                )
        time.sleep(0.4)   # polite API rate — 8 points × 0.4s ≈ 3s total

    if not records:
        log.warning("No records collected.")
        return 0

    df_new = pd.DataFrame(records)
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)

    write_header = not os.path.exists(RAW_PATH)
    df_new.to_csv(RAW_PATH, mode="a", header=write_header, index=False)
    log.info("Appended %d rows → %s", len(records), RAW_PATH)
    return len(records)


if __name__ == "__main__":
    log.info("Snapshot @ %s", datetime.now(timezone.utc).isoformat())
    collect_snapshot()