# %%
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import json
import matplotlib.pyplot as plt
import numpy as np
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
jp_cities = {
    "Fukushima": (37.344608, 140.033492),
    "Kochi": (33.424589, 132.847264),
    "Akita": (39.516758, 140.072186),
    "Hokkaido": (41.961206, 140.585947),
    "Kazaya": (34.045, 135.786667),
    "Owase": (34.136667, 136.193333),
    "Kamkikita": (34.136667, 136.005),
    "Hongawa": (33.765, 133.338333),
    "Yusuhara": (32.056667, 132.921667),
    "Ohguchi": (32.046667, 130.626667),
}

def get_jma_historical(jp_cities: dict[str, tuple[float, float]], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    city_obs = {}
    for city in jp_cities:
        params = {
            "latitude": jp_cities[city][0],
            "longitude": jp_cities[city][1],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "dew_point_2m", "precipitation", "pressure_msl", "cloud_cover_low", "wind_speed_10m", "wind_direction_10m"],
            "wind_speed_unit": "ms"
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        # print(f"Elevation {response.Elevation()} m asl")
        # print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
        # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

        dist = np.sqrt((response.Latitude() - jp_cities[city][0])**2 + (response.Longitude() - jp_cities[city][1])**2)
        print(f"City: {city}, Distance to requested coordinates {dist:.2f}°")

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
        hourly_pressure_msl = hourly.Variables(3).ValuesAsNumpy()
        hourly_cloud_cover_low = hourly.Variables(4).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(5).ValuesAsNumpy()
        hourly_wind_direction_10m = hourly.Variables(6).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}
        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["dew_point_2m"] = hourly_dew_point_2m
        hourly_data["precipitation"] = hourly_precipitation
        hourly_data["pressure_msl"] = hourly_pressure_msl
        hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
        hourly_data["wind_direction_10m"] = hourly_wind_direction_10m

        hourly_dataframe = pd.DataFrame(data = hourly_data)
        city_obs[city] = hourly_dataframe
    return city_obs

# %%
start_date = "2024-01-01"
end_date = "2024-03-31"
winter_obs = get_jma_historical(jp_cities, start_date, end_date)
# %%
# load pointy predictions
with open("/fast/wbhaoxing/deep/evals/cities/jp_pointy_era5_wm_hres.json", "r") as f:
    winter_preds = json.load(f)

# %%
pointy_errors = {city: defaultdict(list) for city in jp_cities}
era5_errors = {city: defaultdict(list) for city in jp_cities}
wm_errors = {city: defaultdict(list) for city in jp_cities}
hres_errors = {city: defaultdict(list) for city in jp_cities}
for city in winter_preds:
    city_pred = winter_preds[city]
    for fcst_date_str, fcst_hour, temp_pointy, temp_era5, temp_wm, temp_hres in city_pred:
        fcst_date = datetime.strptime(fcst_date_str, "%Y%m%d%H")
        obs_date = fcst_date + timedelta(hours=fcst_hour)
        obs = winter_obs[city].loc[winter_obs[city]["date"] == str(obs_date)]
        try:
            temp = obs["temperature_2m"].values[0]
        except:
            print(city, fcst_date, obs_date)
        pointy_errors[city][fcst_hour].append(temp_pointy - temp)
        era5_errors[city][fcst_hour].append(temp_era5 - temp)
        wm_errors[city][fcst_hour].append(temp_wm - temp)
        hres_errors[city][fcst_hour].append(temp_hres - temp)

# %%
fig, ax = plt.subplots(2, 5, figsize=(30, 10), dpi=200)
fig.suptitle("Jan-Feb 2024, JPower points of interest\nValidated against JMA historical weather data", fontweight="bold", fontsize=16)
for i, city in enumerate(jp_cities):
    row, col = i // 5, i % 5
    ax[row, col].plot(pointy_errors[city].keys(), [np.mean(pointy_errors[city][h]) for h in pointy_errors[city]], label="Pointy")
    ax[row, col].plot(wm_errors[city].keys(), [np.mean(wm_errors[city][h]) for h in wm_errors[city]], label="WeatherMesh")
    ax[row, col].plot(hres_errors[city].keys(), [np.nanmean(hres_errors[city][h]) for h in hres_errors[city]], label="IFS HRES")
    ax[row, col].plot(era5_errors[city].keys(), [np.mean(era5_errors[city][h]) for h in era5_errors[city]], label="ERA-5", linestyle="--")
    # dash line at 0
    ax[row, col].axhline(0, color="gray", alpha=0.5)
    ax[row, col].set_title(city)
    ax[row, col].set_xlabel("Forecast hour")
    ax[row, col].set_ylabel("Bias (C)")
    ax[row, col].legend()
# %%
# same but RMSE
fig, ax = plt.subplots(2, 5, figsize=(30, 10), dpi=200)
fig.suptitle("Jan-Feb 2024, JPower points of interest\nValidated against JMA historical weather data", fontweight="bold", fontsize=16)
for i, city in enumerate(jp_cities):
    row, col = i // 5, i % 5
    ax[row, col].plot(pointy_errors[city].keys(), [np.sqrt(np.mean(np.square(pointy_errors[city][h]))) for h in pointy_errors[city]], label="Pointy")
    ax[row, col].plot(wm_errors[city].keys(), [np.sqrt(np.mean(np.square(wm_errors[city][h]))) for h in wm_errors[city]], label="WeatherMesh")
    ax[row, col].plot(hres_errors[city].keys(), [np.sqrt(np.nanmean(np.square(hres_errors[city][h]))) for h in hres_errors[city]], label="IFS HRES")
    ax[row, col].plot(era5_errors[city].keys(), [np.sqrt(np.mean(np.square(era5_errors[city][h]))) for h in era5_errors[city]], label="ERA-5", linestyle="--")
    ax[row, col].set_title(city)
    ax[row, col].set_xlabel("Forecast hour")
    ax[row, col].set_ylabel("RMSE (C)")
    ax[row, col].legend()

# %%
# Summer
summer_obs = get_jma_historical(jp_cities, "2024-07-01", "2024-09-30")
with open("/fast/wbhaoxing/deep/evals/cities/jp_pointy_era5_wm_hres_summer.json", "r") as f:
    summer_preds = json.load(f)

# %%
pointy_errors = {city: defaultdict(list) for city in jp_cities}
era5_errors = {city: defaultdict(list) for city in jp_cities}
wm_errors = {city: defaultdict(list) for city in jp_cities}
hres_errors = {city: defaultdict(list) for city in jp_cities}
for city in summer_preds:
    city_pred = summer_preds[city]
    for fcst_date_str, fcst_hour, temp_pointy, temp_era5, temp_wm, temp_hres in city_pred:
        fcst_date = datetime.strptime(fcst_date_str, "%Y%m%d%H")
        obs_date = fcst_date + timedelta(hours=fcst_hour)
        obs = summer_obs[city].loc[summer_obs[city]["date"] == str(obs_date)]
        try:
            temp = obs["temperature_2m"].values[0]
        except:
            print(city, fcst_date, obs_date)
        pointy_errors[city][fcst_hour].append(temp_pointy - temp)
        era5_errors[city][fcst_hour].append(temp_era5 - temp)
        wm_errors[city][fcst_hour].append(temp_wm - temp)
        hres_errors[city][fcst_hour].append(temp_hres - temp)

# %%
fig, ax = plt.subplots(2, 5, figsize=(30, 10), dpi=200)
fig.suptitle("Jul-Aug 2024, JPower points of interest\nValidated against JMA historical weather data", fontweight="bold", fontsize=16)
for i, city in enumerate(jp_cities):
    row, col = i // 5, i % 5
    ax[row, col].plot(pointy_errors[city].keys(), [np.mean(pointy_errors[city][h]) for h in pointy_errors[city]], label="Pointy")
    ax[row, col].plot(wm_errors[city].keys(), [np.mean(wm_errors[city][h]) for h in wm_errors[city]], label="WeatherMesh")
    ax[row, col].plot(hres_errors[city].keys(), [np.nanmean(hres_errors[city][h]) for h in hres_errors[city]], label="IFS HRES")
    ax[row, col].plot(era5_errors[city].keys(), [np.nanmean(era5_errors[city][h]) for h in era5_errors[city]], label="ERA-5", linestyle="--")
    # dash line at 0
    ax[row, col].axhline(0, color="gray", alpha=0.5)
    ax[row, col].set_title(city)
    ax[row, col].set_xlabel("Forecast hour")
    ax[row, col].set_ylabel("Bias (C)")
    ax[row, col].legend()
# %%
# same but RMSE
fig, ax = plt.subplots(2, 5, figsize=(30, 10), dpi=200)
fig.suptitle("Jul-Aug 2024, JPower points of interest\nValidated against JMA historical weather data", fontweight="bold", fontsize=16)
for i, city in enumerate(jp_cities):
    row, col = i // 5, i % 5
    ax[row, col].plot(pointy_errors[city].keys(), [np.sqrt(np.mean(np.square(pointy_errors[city][h]))) for h in pointy_errors[city]], label="Pointy")
    ax[row, col].plot(wm_errors[city].keys(), [np.sqrt(np.mean(np.square(wm_errors[city][h]))) for h in wm_errors[city]], label="WeatherMesh")
    ax[row, col].plot(hres_errors[city].keys(), [np.sqrt(np.nanmean(np.square(hres_errors[city][h]))) for h in hres_errors[city]], label="IFS HRES")
    ax[row, col].plot(era5_errors[city].keys(), [np.sqrt(np.nanmean(np.square(era5_errors[city][h]))) for h in era5_errors[city]], label="ERA-5", linestyle="--")
    ax[row, col].set_title(city)
    ax[row, col].set_xlabel("Forecast hour")
    ax[row, col].set_ylabel("RMSE (C)")
    ax[row, col].legend()

# %%
