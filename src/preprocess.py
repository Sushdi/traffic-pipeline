"""
Feature engineering pipeline.
Reads data/raw/traffic_raw.csv → writes data/processed/traffic_features.csv
"""

import os
import logging

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

with open("params.yaml") as f:
    CFG = yaml.safe_load(f)

RAW_PATH  = CFG["paths"]["raw"]
PROC_PATH = CFG["paths"]["processed"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, parse_dates=["timestamp"])
    df = df.sort_values(["location_name", "timestamp"]).reset_index(drop=True)
    log.info("Loaded %d rows from %s", len(df), RAW_PATH)
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["current_speed"]         >= 0]
    df = df[df["free_flow_speed"]       >  0]
    df = df[df["current_travel_time"]   >  0]
    df = df[df["free_flow_travel_time"] >  0]
    df = df[df["confidence"]            >= 0.5]
    log.info("Outlier removal: %d → %d rows", before, len(df))
    return df.reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["min_sin"]  = np.sin(2 * np.pi * df["minute"] / 60)
    df["min_cos"]  = np.cos(2 * np.pi * df["minute"] / 60)

    df["is_peak_morning"] = ((df["hour"] >= 7)  & (df["hour"] <= 9)).astype(int)
    df["is_peak_evening"] = ((df["hour"] >= 16) & (df["hour"] <= 19)).astype(int)
    df["is_off_peak"]     = (
        ~((df["hour"] >= 7) & (df["hour"] <= 9)) &
        ~((df["hour"] >= 16) & (df["hour"] <= 19))
    ).astype(int)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    result_parts = []

    for loc_name, grp in df.groupby("location_name"):
        grp = grp.copy()

        # Speed lags
        for lag, col in [(1, "speed_lag_1"), (3, "speed_lag_3"),
                         (6, "speed_lag_6"), (12, "speed_lag_12")]:
            grp[col] = grp["current_speed"].shift(lag)

        # Speed ratio lags
        for lag, col in [(1, "ratio_lag_1"), (3, "ratio_lag_3"), (6, "ratio_lag_6")]:
            grp[col] = grp["speed_ratio"].shift(lag)

        # Rolling features
        grp["speed_roll_mean_6"]  = grp["current_speed"].rolling(6,  min_periods=1).mean()
        grp["speed_roll_mean_12"] = grp["current_speed"].rolling(12, min_periods=1).mean()
        grp["speed_roll_std_6"]   = grp["current_speed"].rolling(6,  min_periods=1).std().fillna(0)
        grp["tt_ratio_roll_6"]    = grp["travel_time_ratio"].rolling(6, min_periods=1).mean()

        # Speed trend
        grp["speed_trend"] = grp["current_speed"] - grp["speed_roll_mean_6"]

        result_parts.append(grp)

    df = pd.concat(result_parts).sort_values(["location_name", "timestamp"])

    before = len(df)
    df = df.dropna(subset=["speed_lag_12", "ratio_lag_6"])
    log.info("Dropped %d warm-up rows (lag NaNs). %d remain.", before - len(df), len(df))
    return df.reset_index(drop=True)


def encode_frc(df: pd.DataFrame) -> pd.DataFrame:
    frc_map = {"FRC0": 0, "FRC1": 1, "FRC2": 2, "FRC3": 3,
               "FRC4": 4, "FRC5": 5, "FRC6": 6, "FRC7": 7}
    df["frc_code"] = df["frc"].map(frc_map).fillna(4).astype(int)
    return df


def run():
    df = load_raw()
    df = remove_outliers(df)
    df = add_time_features(df)
    df = add_lag_and_rolling_features(df)
    df = encode_frc(df)

    log.info("Class distribution:\n%s", df["congestion_level"].value_counts())

    os.makedirs(os.path.dirname(PROC_PATH), exist_ok=True)
    df.to_csv(PROC_PATH, index=False)
    log.info("Saved %d rows → %s", len(df), PROC_PATH)


if __name__ == "__main__":
    run()