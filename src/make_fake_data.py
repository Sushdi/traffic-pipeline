import pandas as pd
import numpy as np
import os

n = 100
df = pd.DataFrame({
    "hour_sin": np.random.uniform(-1,1,n),
    "hour_cos": np.random.uniform(-1,1,n),
    "dow_sin": np.random.uniform(-1,1,n),
    "dow_cos": np.random.uniform(-1,1,n),
    "min_sin": np.random.uniform(-1,1,n),
    "min_cos": np.random.uniform(-1,1,n),
    "is_weekend": np.random.randint(0,2,n),
    "is_peak_morning": np.random.randint(0,2,n),
    "is_peak_evening": np.random.randint(0,2,n),
    "current_speed": np.random.uniform(10,100,n),
    "free_flow_speed": np.random.uniform(50,120,n),
    "current_travel_time": np.random.uniform(60,600,n),
    "free_flow_travel_time": np.random.uniform(60,300,n),
    "speed_ratio": np.random.uniform(0.3,1.0,n),
    "travel_time_ratio": np.random.uniform(1.0,3.0,n),
    "confidence": np.random.uniform(0.5,1.0,n),
    "road_closure": np.random.randint(0,2,n),
    "frc_code": np.random.randint(0,8,n),
    "speed_lag_1": np.random.uniform(10,100,n),
    "speed_lag_3": np.random.uniform(10,100,n),
    "speed_lag_6": np.random.uniform(10,100,n),
    "speed_lag_12": np.random.uniform(10,100,n),
    "ratio_lag_1": np.random.uniform(0.3,1.0,n),
    "ratio_lag_3": np.random.uniform(0.3,1.0,n),
    "ratio_lag_6": np.random.uniform(0.3,1.0,n),
    "speed_roll_mean_6": np.random.uniform(10,100,n),
    "speed_roll_mean_12": np.random.uniform(10,100,n),
    "speed_roll_std_6": np.random.uniform(0,20,n),
    "tt_ratio_roll_6": np.random.uniform(1.0,3.0,n),
    "speed_trend": np.random.uniform(-20,20,n),
    "congestion_level": np.random.choice(["free_flow","moderate","congested"],n),
})

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/traffic_features.csv", index=False)
print(f"Created {n} fake rows for testing")
print(df["congestion_level"].value_counts())