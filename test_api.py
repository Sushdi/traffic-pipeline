import requests

API_KEY = "YOUR_API_KEY"

lat = 28.6139   # Delhi example
lon = 77.2090

url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={API_KEY}"

response = requests.get(url)
data = response.json()

print(data)