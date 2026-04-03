import requests
import pandas as pd
from datetime import datetime

API_KEY = "csBWBnfOPKvqTx5bIHgPqEpPf5RQjOnR"

lat = 28.6139
lon = 77.2090

url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={API_KEY}"

response = requests.get(url)
data = response.json()

#Extract features
features = {
    "timestamp": datetime.now(),
    "latitude": lat,
    "longitude": lon,
    "current_speed": data["flowSegmentData"]["currentSpeed"],
    "free_flow_speed": data["flowSegmentData"]["freeFlowSpeed"],
    "confidence": data["flowSegmentData"]["confidence"],
    "road_closure": data["flowSegmentData"]["roadClosure"],
    "current_travel_time": data["flowSegmentData"]["currentTravelTime"],
    "free_flow_travel_time": data["flowSegmentData"]["freeFlowTravelTime"]
}

#Function to calculate congestion
def get_congestion(row):
    ratio = row["current_speed"] / row["free_flow_speed"]
    
    if ratio > 0.75:
        return "Low"
    elif ratio > 0.4:
        return "Medium"
    else:
        return "High"

#Convert to DataFrame
df = pd.DataFrame([features])

#Add derived features
df["speed_ratio"] = df["current_speed"] / df["free_flow_speed"]
df["delay"] = df["current_travel_time"] - df["free_flow_travel_time"]

#Adding target column
df["congestion"] = df.apply(get_congestion, axis=1)

print(df)