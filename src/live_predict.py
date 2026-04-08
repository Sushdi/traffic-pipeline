"""
Fetches live TomTom data for all Munich waypoints and sends to /predict.
Run while the API is running on 127.0.0.1:8000
"""

import os
import time
import math
import requests
import yaml
from datetime import datetime, timezone

with open("params.yaml") as f:
    CFG = yaml.safe_load(f)

API_KEY     = os.environ["TOMTOM_API_KEY"]
WAYPOINTS   = CFG["waypoints"]
PREDICT_URL = "http://127.0.0.1:8000/predict"


def fetch_tomtom(lat, lon):
    url = (
        "https://api.tomtom.com/traffic/services/4"
        f"/flowSegmentData/absolute/10/json"
        f"?point={lat},{lon}&key={API_KEY}"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()["flowSegmentData"]


def build_features(fsd: dict, history: list) -> dict:
    now    = datetime.now(timezone.utc)
    hour   = now.hour
    minute = now.minute
    dow    = now.weekday()

    cur_speed  = float(fsd["currentSpeed"])
    free_speed = float(fsd["freeFlowSpeed"])
    cur_tt     = float(fsd["currentTravelTime"])
    free_tt    = float(fsd["freeFlowTravelTime"])

    frc_map  = {"FRC0":0,"FRC1":1,"FRC2":2,"FRC3":3,
                "FRC4":4,"FRC5":5,"FRC6":6,"FRC7":7}
    frc_code = frc_map.get(fsd.get("frc",""), 4)

    speeds = [h["currentSpeed"] for h in history] + [cur_speed]
    speeds = speeds[-13:]

    def lag(n):
        return speeds[-(n+1)] if len(speeds) > n else cur_speed

    def roll_mean(n):
        s = speeds[-n:]
        return sum(s) / len(s)

    def roll_std(n):
        s = speeds[-n:]
        if len(s) < 2:
            return 0.0
        mean = sum(s) / len(s)
        return (sum((x - mean)**2 for x in s) / len(s)) ** 0.5

    tt_ratios = (
        [h["currentTravelTime"] / max(h["freeFlowTravelTime"], 1)
         for h in history[-6:]]
        + [cur_tt / max(free_tt, 1)]
    )

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
        "speed_lag_1":  lag(1),
        "speed_lag_3":  lag(3),
        "speed_lag_6":  lag(6),
        "speed_lag_12": lag(12),
        "speed_roll_mean_6":  roll_mean(6),
        "speed_roll_mean_12": roll_mean(12),
        "speed_roll_std_6":   roll_std(6),
        "tt_ratio_roll_6":    sum(tt_ratios[-6:]) / min(6, len(tt_ratios)),
        "speed_trend":        cur_speed - roll_mean(6),
    }


def main():
    history = {wp["name"]: [] for wp in WAYPOINTS}

    print(f"\n{'\n'}")
    print(f"  Live Munich Traffic — {datetime.now().strftime('%A %H:%M:%S')}")
    print(f"{'\n'}")
    print(f"{'Location':<22} {'Speed':>6} {'Free':>6} {'Prediction':<12} {'Conf':>6}")
    print(f"{'----\n'}")

    for wp in WAYPOINTS:
        try:
            fsd  = fetch_tomtom(wp["lat"], wp["lon"])
            feat = build_features(fsd, history[wp["name"]])
            history[wp["name"]].append(fsd)

            resp = requests.post(PREDICT_URL, json=feat, timeout=5)
            pred = resp.json()

            label = pred["congestion_level"]
            probs = pred["probabilities"]
            conf  = max(probs.values())
            icon  = {"free_flow": "🟢", "moderate": "🟡",
                     "congested": "🔴"}.get(label, "⚪")

            print(
                f"{wp['name']:<22} "
                f"{fsd['currentSpeed']:>5}k "
                f"{fsd['freeFlowSpeed']:>5}k "
                f"{icon} {label:<10} "
                f"{conf*100:>5.1f}%"
            )
            time.sleep(0.3)

        except Exception as e:
            print(f"{wp['name']:<22} ERROR: {e}")

    print(f"{'----'}\n")


if __name__ == "__main__":
    main()