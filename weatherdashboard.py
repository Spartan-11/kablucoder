#Weatherdashboard

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------- CONFIGURATION --------
API_KEY =  "e7468d169ef270e633c2f0489e9e92e3" # Replace with your OpenWeatherMap API key
CITY = "bokaro"
UNITS = "metric"  # use 'imperial' for Fahrenheit
API_URL = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units={UNITS}"

# -------- FETCH DATA --------
response = requests.get(API_URL)
data = response.json()

# Check if response is valid
if data['cod'] != '200':
    print("Failed to fetch data:", data['message'])
    exit()

# -------- PROCESS DATA --------
forecast_list = data['list']
weather_data = {
    "datetime": [entry["dt_txt"] for entry in forecast_list],
    "temperature": [entry["main"]["temp"] for entry in forecast_list],
    "humidity": [entry["main"]["humidity"] for entry in forecast_list],
    "weather": [entry["weather"][0]["main"] for entry in forecast_list],
}

df = pd.DataFrame(weather_data)
df["datetime"] = pd.to_datetime(df["datetime"])

# -------- VISUALIZATION --------
sns.set(style="whitegrid")
plt.figure(figsize=(14, 10))

# Temperature trend
plt.subplot(3, 1, 1)
sns.lineplot(data=df, x="datetime", y="temperature", marker="o", color="tomato")
plt.title(f"Temperature Forecast in {CITY}")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)

# Humidity trend
plt.subplot(3, 1, 2)
sns.lineplot(data=df, x="datetime", y="humidity", marker="o", color="dodgerblue")
plt.title(f"Humidity Forecast in {CITY}")
plt.ylabel("Humidity (%)")
plt.xticks(rotation=45)

# Weather condition count (bar chart)
plt.subplot(3, 1, 3)
sns.countplot(data=df, x="weather", palette="Set2")
plt.title(f"Weather Condition Frequency in Forecast")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
