# %%
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('/fast/haoxing/windborne')
from meteo.tools.process_dataset import download_s3_file
import torch
from tqdm import tqdm
np.set_printoptions(precision=4, suppress=True)

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from datetime import datetime, timezone, timedelta
from PIL import Image
import pickle
import pygrib
import requests
from scipy.interpolate import RegularGridInterpolator

# %%
metar = np.load("/fast/proc/metar/metar_2022_8_1.npy")
metar_keys = ('station', 'valid', 'lon', 'lat', 'elevation', 'tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'p01i', 'alti', 'mslp', 'vsby', 'gust', 'skyl1', 'skyl2', 'skyl3', 'skyl4', 'ice_accretion_1hr', 'ice_accretion_3hr', 'ice_accretion_6hr', 'peak_wind_gust', 'peak_wind_drct', 'peak_wind_time', 'feel', 'snowdepth')
# %%
metar[:10]
# %%
stations = pickle.load(open("/fast/proc/metar/stations.pickle", "rb"))
# %%
len(stations)
# %%
boston1 = [42.412917, -71.158018]
boston2 = [42.293701, -70.940969]

for station, coords in stations.items():
    if boston2[0] < coords[0] < boston1[0] and boston1[1] < coords[1] < boston2[1]:
        print(station, coords)
# %%
# "big"
cities = {
    "Atlanta, GA": b"ATL",
    "Boston, MA": b"BOS",
    "Chicago, IL": b"ORD",
    "Detroit, MI": b"DTW",
    "Honolulu, HI": b"PHNL",
    "Houston, TX": b"IAH",
    "Miami, FL": b"MIA",
    "New York, NY": b"LGA",
    "Philadelphia, PA": b"PHL",
    "San Jose, CA": b"SJC",
    "Seattle, WA": b"SEA",
    "Barcelona, Spain": b"LEBL",
    "Beijing, China": b"ZBAA",
    "Cairo, Egypt": b"HECA",
    "Dubai, UAE": b"OMDB",
    "Espargos, Cape Verde": b"GVAC",
    "Istanbul, Turkey": b"LTBA",
    "Johannesburg, South Africa": b"FAOR",
    "London, UK": b"EGLL",
    "Mexico City, Mexico": b"MMMX",
    "Moscow, Russia": b"UUEE",
    "Mumbai, India": b"VABB",
    "Nairobi, Kenya": b"HKJK",
    "Sao Paolo, Brazil": b"SBGR",
    "Seoul, South Korea": b"RKSI",
    "Sydney, Australia": b"YSSY",
    "Tamuning, Guam": b"PGUM",
    "Tokyo, Japan": b"RJTT",
    "Toronto, Canada": b"CYYZ",
    "Vienna, Austria": b"LOWW",
}

# final list
cities = {
    "Atlanta, GA": b"ATL",
    "Boston, MA": b"BOS",
    "Chicago, IL": b"ORD",
    "Houston, TX": b"IAH",
    "Miami, FL": b"MIA",
    "New York, NY": b"LGA",
    "San Jose, CA": b"SJC",
    "Barcelona, Spain": b"LEBL",
    "Beijing, China": b"ZBAA",
    "London, UK": b"EGLL",
    "Mumbai, India": b"VABB",
    "Nairobi, Kenya": b"HKJK",
    "Sao Paolo, Brazil": b"SBGR",
    "Seoul, South Korea": b"RKSI",
    "Toronto, Canada": b"CYYZ",
}

cities = {
    "São Paulo": b"SBGR",
    "Rio de Janeiro": b"SBGL",
    "Belo Horizonte": b"SBCF",
    "Brasília": b"SBBR",
    "Salvador": b"SBSV",
    "Fortaleza": b"SBFZ",
    "Recife": b"SBRF",
    "Manaus": b"SBEG",
    "Curitiba": b"SBCT",
    #"Porto Alegre": b"SBPA",
    "Belem": b"SBBE",
}

reverse_cities = {v.decode(): k for k, v in cities.items()}


city_coords = [stations[city] if city in stations else None for city in cities.values() ]
city_coords = np.array(city_coords)
city_coords = city_coords[:, [1, 0]]
city_coords

# %%
# plot the city locations on US map
fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=200, subplot_kw={'projection': ccrs.PlateCarree()})
plt.scatter(city_coords[:, 0], city_coords[:, 1], c='r', s=10)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.STATES, linestyle=':')
plt.show()

# %%
city_timestamps = defaultdict(list)
for entry in metar:
    if entry[0] in cities.values():
        city_timestamps[entry[0]].append(entry[1])

# %%
def get_closest_utc_hour(timestamp: int) -> int:
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    if dt.minute < 30:
        return dt.hour
    else:
        return (dt + timedelta(hours=1)).hour

def get_closest_utc_whole_hour(timestamp: int) -> datetime:
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    if dt.minute < 30:
        return dt.replace(minute=0, second=0)
    else:
        return (dt + timedelta(hours=1)).replace(minute=0, second=0)

get_closest_utc_hour(city_timestamps[b"LAX"][0])
# %%
[datetime.fromtimestamp(t) for t in city_timestamps[b"LAX"]]

# %%
# get all the METAR targets

# for date in date range
# open METAR file
# for entry in file
# if entry in cities
# check if entry is 0z
# get observation

city_obs = defaultdict(list)

for day in tqdm(range(1, 32)):
    for month in [7, 8]:
        metar = np.load(f"/fast/proc/metar/metar_2024_{month}_{day}.npy")
        for entry in metar:
            if entry[0] in cities.values():
                utc_dt = get_closest_utc_whole_hour(entry[1])
                # if utc_dt.hour == 0:
                city_obs[utc_dt.timestamp()].append(entry)

for day in tqdm(range(1, 20)):
    metar = np.load(f"/fast/proc/metar/metar_2024_9_{day}.npy")
    for entry in metar:
        if entry[0] in cities.values():
            utc_dt = get_closest_utc_whole_hour(entry[1])
            # if utc_dt.hour == 0:
            city_obs[utc_dt.timestamp()].append(entry)

# %%
# %%
meta = json.load(open("/huge/deep/realtime/outputs/yamahabachelor/meta.vGM0.json", "r"))
wm_lats, wm_lons = meta["lats"], meta["lons"]
wm_lons = wm_lons[720:] + wm_lons[:720]
os.makedirs("/huge/deep/realtime/outputs/WeatherMesh-backtest/2022070100/det", exist_ok=True)
download_s3_file("wb-dlnwp", f"WeatherMesh-backtest/2022070100/det/24.vGM0.npy", f"/huge/deep/realtime/outputs/WeatherMesh-backtest/2022070100/det/24.vGM0.npy")
wm_out = np.load(f"/huge/deep/realtime/outputs/WeatherMesh-backtest/2022070100/det/24.vGM0.npy")
wm_out = np.concatenate([wm_out[:,720:], wm_out[:,:720]], axis=1)
# %%
tempk_interp = RegularGridInterpolator((wm_lats, wm_lons), wm_out[:,:,-5], method="linear")

# %%
plt.imshow(wm_out[:,:,-5])

# %%
tempk_interp([-71.0097, 42.3606])
# %%
# for init date
# for forecast hour
# download WeatherMesh output
city_errors = defaultdict(list)

for month in [7, 8]:
    for day in tqdm(range(1, 32)):
        init_datestr = f"2024{month:02d}{day:02d}00"
        for fh in range(24, 337, 24):
            print(f"processing {init_datestr} forecast hour {fh}")
            obs_date = datetime(2024, month, day, 0, tzinfo=timezone.utc) + timedelta(hours=fh)
            os.makedirs(f"/huge/deep/realtime/outputs/WeatherMesh-backtest/{init_datestr}/det", exist_ok=True)
            download_s3_file("wb-dlnwp", f"WeatherMesh-backtest/{init_datestr}/det/{fh}.vGM0.npy", f"/huge/deep/realtime/outputs/WeatherMesh-backtest/{init_datestr}/det/{fh}.vGM0.npy")
            wm_out = np.load(f"/huge/deep/realtime/outputs/WeatherMesh-backtest/{init_datestr}/det/{fh}.vGM0.npy")
            wm_out = np.concatenate([wm_out[:,720:], wm_out[:,:720]], axis=1)
            tempk_interp = RegularGridInterpolator((wm_lats, wm_lons), wm_out[:,:,-5], method="linear")
            for entry in city_obs[obs_date.timestamp()]:
                lon, lat, tempf = entry[2], entry[3], entry[5]
                tempk = (tempf - 32) * 5 / 9 + 273.15
                tempk_wm = tempk_interp([lat, lon])[0]
                city_errors[entry[0].decode()].append((init_datestr, fh, tempk, tempk_wm, tempk - tempk_wm))

# %%
city_errors["ATL"] = list(set(city_errors["ATL"]))
# %%
# save city_errors
with open("/fast/wbhaoxing/deep/evals/city_errors.json", "w") as f:
    json.dump(city_errors, f)
# %%
import pandas as pd
# turn city_errors into a dataframe
city_errors_df = pd.DataFrame(columns=["station", "init_date", "forecast_hour", "tempk_obs", "tempk_wm", "error"])
for city, errors in city_errors.items():
    city_errors_df = pd.concat([city_errors_df, pd.DataFrame([city] + list(errors), columns=["station", "init_date", "forecast_hour", "tempk_obs", "tempk_wm", "error"])])
# %%
# compute RMSE for each city by forecast hour
city_rmse = defaultdict(list)
for city, errors in city_errors.items():
    for fh in range(24, 336, 24):
        errors_fh = [error[4] for error in errors if error[1] == fh and not np.isnan(error[4])]
        rmse = np.sqrt(np.mean(np.square(errors_fh)))
        city_rmse[city].append(rmse)
# %%
city_rmse

# %%
fig = plt.figure(dpi=300)
for city in city_rmse.keys():
    if city not in [v.decode() for v in cities.values()]:
        continue
    plt.plot(range(1,14), city_rmse[city], label=city)
plt.xlabel("Forecast lead time (days)")
plt.xticks([1,3,7,10,14])
plt.ylabel("RMSE (K)")
plt.title("WM")
plt.grid()
plt.legend()
# %%
len(test["ATL"])
# %%
sfo_10day_tempk_obs = [error[3] for error in test["SFO"] if error[1] == 240]
sea_10day_tempk_obs = [error[3] for error in test["SEA"] if error[1] == 240]

# %%
plt.hist(sfo_10day_tempk_obs, bins=50, alpha=0.5, label="SFO")
plt.hist([error[3] for error in test["MIA"] if error[1] == 240], bins=50, alpha=0.5, label="MIA")
plt.hist(sea_10day_tempk_obs, bins=50, alpha=0.5, label="SEA")
plt.hist([error[3] for error in test["PHL"] if error[1] == 240], bins=50, alpha=0.5, label="PHL")
plt.legend()

# %%
mars = "/fast/ignored/mars_data/output_2024012006.grib"
grbs = pygrib.open(mars)
# get 2m temperature
grb = grbs.select(name="2 metre temperature")[0]
data = grb.values
# %%
city_hres_errors = defaultdict(list)

for month in [7, 8]:
    for day in tqdm(range(1, 32)):
        init_datestr = f"2024{month:02d}{day:02d}00"
        for fh in list(range(24, 10*24, 24)) + [234]:
            print(f"processing {init_datestr} forecast hour {fh}")
            obs_date = datetime(2024, month, day, 0, tzinfo=timezone.utc) + timedelta(hours=fh)
            os.makedirs(f"/huge/proc/weather-archive/ecmwf/{init_datestr[:-2]}/00z/ifs/0p25/oper", exist_ok=True)
            download_s3_file("wb-weather-archive", f"ecmwf/{init_datestr[:-2]}/00z/ifs/0p25/oper/{init_datestr}0000-{fh}h-oper-fc.grib2", f"/huge/proc/weather-archive/ecmwf/{init_datestr[:-2]}/00z/ifs/0p25/oper/{init_datestr}0000-{fh}h-oper-fc.grib2")
            grbs = pygrib.open(f"/huge/proc/weather-archive/ecmwf/{init_datestr[:-2]}/00z/ifs/0p25/oper/{init_datestr}0000-{fh}h-oper-fc.grib2")
            hres_out = grbs.select(name="2 metre temperature")[0].values[:720,:]
            tempk_interp = RegularGridInterpolator((wm_lats, wm_lons), hres_out, method="linear")
            for entry in city_obs[obs_date.timestamp()]:
                lon, lat, tempf = entry[2], entry[3], entry[5]
                tempk = (tempf - 32) * 5 / 9 + 273.15
                tempk_hres = tempk_interp([lat, lon])[0]
                city_hres_errors[entry[0].decode()].append((init_datestr, fh, tempk, tempk_hres, tempk - tempk_hres))
        with open("/fast/wbhaoxing/deep/evals/city_hres_errors.json", "w") as f:
            json.dump(city_hres_errors, f)
# %%
server = 'https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com'
cycle_time = datetime(2024, 2, 1, 0)
forecast_hour = 240
key = f"{cycle_time.strftime('%Y%m%d/%Hz')}/ifs/0p25/oper/{cycle_time.strftime('%Y%m%d%H')}0000-{forecast_hour}h-oper-fc.grib2"
url = f"{server}/{key}"
urls = [url]
r = requests.get(url, stream=True)
r

# %%
city_hres_errors
# %%
plt.imshow(hres_out)
# %%
# %%
with open("/fast/wbhaoxing/deep/evals/city_hres_errors_brazil_summer.json", "r") as f:
    city_hres_errors = json.load(f)

# %%
city_hres_errors_dedup = defaultdict(list)
for city, errors in city_hres_errors.items():
    error_tuples = [tuple(error) for error in errors]
    error_tuples = list(set(error_tuples))
    print(f"{city}: {len(errors)} -> {len(error_tuples)}, number of duplicates: {len(errors) - len(error_tuples)}")
    city_hres_errors_dedup[city] = error_tuples
# %%
# compute RMSE for each city by forecast hour
city_hres_rmse = defaultdict(list)
for city, errors in city_hres_errors.items():
    for fh in list(range(24, 240, 24)) + [234]:
        errors_fh = [error[4] for error in errors if error[1] == fh and not np.isnan(error[4])]
        rmse = np.sqrt(np.mean(np.square(errors_fh)))
        city_hres_rmse[city].append(float(rmse))
city_hres_rmse

# %%
city_hres_rmse_dedup = defaultdict(list)
for city, errors in city_hres_errors_dedup.items():
    #for fh in list(range(24, 240, 24)) + [234]:
    for fh in list(range(24, 241, 24)):
        errors_fh = [error[4] for error in errors if error[1] == fh and not np.isnan(error[4])]
        rmse = np.sqrt(np.mean(np.square(errors_fh)))
        city_hres_rmse_dedup[city].append(float(rmse))
city_hres_rmse_dedup

# %%
fig = plt.figure(dpi=300)
for city in city_hres_rmse.keys():
    if city not in [v.decode() for v in cities.values()]:
        continue
    plt.plot(range(1,11), city_hres_rmse[city], label=city)
plt.xlabel("Forecast lead time (days)")
plt.xticks([1,3,7,10,14])
plt.ylabel("RMSE (K)")
plt.title("IFS")
plt.grid()
plt.legend()
# %%
with open("/fast/wbhaoxing/deep/evals/city_hres_rmse_brazil_summer.json", "w") as f:
    json.dump(city_hres_rmse_dedup, f)

# %%
with open("/fast/wbhaoxing/deep/evals/city_wm_errors_brazil_summer.json", "r") as f:
    city_errors = json.load(f)


# %%
# dedup
city_errors_dedup = defaultdict(list)
for city, errors in city_errors.items():
    error_tuples = [tuple(error) for error in errors]
    error_tuples = list(set(error_tuples))
    print(f"{city}: {len(errors)} -> {len(error_tuples)}, number of duplicates: {len(errors) - len(error_tuples)}")
    city_errors_dedup[city] = error_tuples
# %%
# compute RMSE for each city by forecast hour
city_rmse = defaultdict(list)
for city, errors in city_errors.items():
    for fh in list(range(24, 337, 24)):
        errors_fh = [error[4] for error in errors if error[1] == fh and not np.isnan(error[4])]
        rmse = np.sqrt(np.mean(np.square(errors_fh)))
        city_rmse[city].append(float(rmse))
city_rmse

# %%
city_rmse_dedup = defaultdict(list)
for city, errors in city_errors_dedup.items():
    for fh in list(range(24, 337, 24)):
        errors_fh = [error[4] for error in errors if error[1] == fh and not np.isnan(error[4])]
        rmse = np.sqrt(np.mean(np.square(errors_fh)))
        city_rmse_dedup[city].append(float(rmse))
city_rmse_dedup

# %%
with open("/fast/wbhaoxing/deep/evals/city_wm_rmse_brazil_summer.json", "w") as f:
    json.dump(city_rmse_dedup, f)

# %%
# with open("/fast/wbhaoxing/deep/evals/cities/city_hres_rmse_winter.json", "r") as f:
#     city_hres_rmse = json.load(f)
# with open("/fast/wbhaoxing/deep/evals/cities/city_wm_rmse_winter.json", "r") as f:
#     city_rmse = json.load(f)
with open("/fast/wbhaoxing/deep/evals/city_hres_rmse_big.json", "r") as f:
    city_hres_rmse = json.load(f)
with open("/fast/wbhaoxing/deep/evals/city_wm_rmse_big.json", "r") as f:
    city_rmse = json.load(f)

# %%
# remove empty keys
city_rmse_dedup = {k: v for k, v in city_rmse_dedup.items() if v}
city_hres_rmse_dedup = {k: v for k, v in city_hres_rmse_dedup.items() if v}
# %%
# make table of RMSE comparison
#row_labels = sorted(cities.keys())
row_labels = list(cities.keys())
hres_rmses = np.array([city_hres_rmse_dedup[cities[city].decode()] for city in row_labels])
wm_rmses = np.array([city_rmse_dedup[cities[city].decode()] for city in row_labels])
col_labels = [1,3,5,7,10,"14*"]
hres_rmses = hres_rmses[:,np.array([0,2,4,6,9,9])] * 9/5
wm_rmses = wm_rmses[:,np.array([0,2,4,6,9,13])] * 9/5

fig, ax = plt.subplots(figsize=(8, 14), dpi=300)

ax.set_xticks(np.arange(len(col_labels)))
ax.set_yticks(np.arange(len(row_labels)))
ax.set_xticklabels(col_labels)
ax.set_yticklabels(row_labels)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(len(row_labels)):
    for j in range(len(col_labels)):
        text = ax.text(j, i, f"{wm_rmses[i, j]:.1f} | {hres_rmses[i, j]:.1f}", ha="center", va="center", color="black")

cax = ax.imshow(wm_rmses - hres_rmses, cmap='RdBu_r', alpha=0.8, vmin=-4, vmax=4, aspect=0.5)
cbar = fig.colorbar(cax, orientation='horizontal', pad=0.1, fraction=0.02)
cbar.ax.set_xticks([])
cbar.set_label("WeatherMesh 2.0 better                                            IFS better")

#ax.set_title("2m Temperature Forecasts Verified Against\nJuly-August 2024 METAR Observations\n", fontweight="bold")
ax.set_title("2m Temperature Forecasts Verified Against\nJuly-August 2024 METAR Observations\n", fontweight="bold")
secondary_x = ax.secondary_xaxis('top')
secondary_x.set_xlabel("WeatherMesh 2.0 RMSE (F) | ECMWF IFS RMSE (F)")
secondary_x.set_xticks([])
ax.set_xlabel("Forecast Lead Time (Days)")

fig.text(0.9, 0.06, "* WeatherMesh 14-day RMSE compared with IFS 10-day RMSE", ha='right', fontsize=9)

#plt.tight_layout()
plt.show()

# %%
import csv

with open("/fast/wbhaoxing/deep/evals/cities/cities_summer_wm_vs_hres.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["City", "wm_1", "hres_1", "wm_3", "hres_3", "wm_5", "hres_5", "wm_7", "hres_7", "wm_10", "hres_10", "wm_14", "hres_10"])
    for city, rmses in city_hres_rmse.items():
        if city not in reverse_cities:
            continue
        row = [reverse_cities[city]]
        for i in [0,2,4,6,9]:
            row.append(city_rmse[city][i] * 9/5)
            row.append(city_hres_rmse[city][i] * 9/5)
        row.append(city_rmse[city][13] * 9/5)
        row.append(city_hres_rmse[city][9] * 9/5)
        writer.writerow(row)
# %%
ifs_grib = "/fast/ignored/mars_data/sfc_2024010100_24h.grib"
grbs = pygrib.open(ifs_grib)

idate = datetime(2024, 1, 1, 0)
dates = [idate + timedelta(hours=i) for i in range(0, 24*60, 24)]
fhs = [24 * i for i in range(1, 11)]

i = 0
for grb in grbs:
    if "temperature" in grb.name.lower():
        print(grb)
        date = dates[i//10]
        fh = fhs[i % 10]
        print(date, fh)
        np.save(f"/huge/users/haoxing/ifs/temp2m_{date.strftime('%Y%m%d%H')}_{fh}.npy", grb.values)
        i += 1
# %%   
# %%
all_temps = []
for grb in grbs:
    if "temperature" in grb.name.lower():
        all_temps.append(grb.values)

all_temps = np.array(all_temps)
all_temps.shape

# %%
for i in range(600):
    date = dates[i//10]
    fh = fhs[i % 10]
    np.save(f"/huge/users/haoxing/ifs/temp2m_{date.strftime('%Y%m%d%H')}_{fh}.npy", all_temps[i])
# %%
some_temp = np.load("/huge/users/haoxing/ifs/temp2m_2024010100_24.npy")
# %%
plt.imshow(some_temp)
# %%
(some_temp - 273.15)[300:350, 300:350]
# %%
## WINTER

print("Gathering city observations")
city_obs = defaultdict(list)

for day in tqdm(range(1, 32)):
    for month in [1, 2]:
        if month == 2 and day > 29:
            continue
        metar = np.load(f"/fast/proc/metar/metar_2024_{month}_{day}.npy")
        for entry in metar:
            if entry[0] in cities.values():
                utc_dt = get_closest_utc_whole_hour(entry[1])
                # if utc_dt.hour == 0:
                city_obs[utc_dt.timestamp()].append(entry)

for day in tqdm(range(1, 20)):
    metar = np.load(f"/fast/proc/metar/metar_2024_3_{day}.npy")
    for entry in metar:
        if entry[0] in cities.values():
            utc_dt = get_closest_utc_whole_hour(entry[1])
            # if utc_dt.hour == 0:
            city_obs[utc_dt.timestamp()].append(entry)

# %%
# compute IFS errors on the city observations
print("Computing IFS errors")
city_hres_errors = defaultdict(list)

for month in [1, 2]:
    for day in tqdm(range(1, 32)):
        if month == 2 and day > 29:
            continue
        init_datestr = f"2024{month:02d}{day:02d}00"
        for fh in list(range(24, 10*24+1, 24)):
            print(f"processing {init_datestr} forecast hour {fh}")
            obs_date = datetime(2024, month, day, 0, tzinfo=timezone.utc) + timedelta(hours=fh)
            # os.makedirs(f"/huge/proc/weather-archive/ecmwf/{init_datestr[:-2]}/00z/ifs/0p25/oper", exist_ok=True)
            # download_s3_file("wb-weather-archive", f"ecmwf/{init_datestr[:-2]}/00z/ifs/0p25/oper/{init_datestr}0000-{fh}h-oper-fc.grib2", f"/huge/proc/weather-archive/ecmwf/{init_datestr[:-2]}/00z/ifs/0p25/oper/{init_datestr}0000-{fh}h-oper-fc.grib2")
            # grbs = pygrib.open(f"/huge/proc/weather-archive/ecmwf/{init_datestr[:-2]}/00z/ifs/0p25/oper/{init_datestr}0000-{fh}h-oper-fc.grib2")
            # hres_out = grbs.select(name="2 metre temperature")[0].values[:720,:]
            hres_out = np.load(f"/huge/users/haoxing/ifs/temp2m_{init_datestr}_{fh}.npy")[:720]
            hres_out = np.concatenate([hres_out[:,720:], hres_out[:,:720]], axis=1)
            tempk_interp = RegularGridInterpolator((wm_lats, wm_lons), hres_out, method="linear")
            for entry in city_obs[obs_date.timestamp()]:
                lon, lat, tempf = entry[2], entry[3], entry[5]
                tempk = (tempf - 32) * 5 / 9 + 273.15
                tempk_hres = tempk_interp([lat, lon])[0]
                city_hres_errors[entry[0].decode()].append((init_datestr, fh, tempk, tempk_hres, tempk - tempk_hres))
        with open("/fast/wbhaoxing/deep/evals/city_hres_errors_winter.json", "w") as f:
            json.dump(city_hres_errors, f)
# %%
plt.imshow(hres_out)
# %%
grbs = pygrib.open(f"/huge/proc/weather-archive/ecmwf/20240701/00z/ifs/0p25/oper/20240701000000-24h-oper-fc.grib2")
hres_out_s3 = grbs.select(name="2 metre temperature")[0].values[:720,:]
# %%
plt.imshow(hres_out_s3)
# %%
levels = set()
for grb in grbs:
    levels.add(grb.level)
    if "precip" in grb.name.lower():
        print(grb)
# %%
len(levels)
# %%
os.makedirs("/huge/proc/weather-archive/ecmwf/20240301/00z/ifs/0p25/oper", exist_ok=True)
download_s3_file("wb-weather-archive", "ecmwf/20240301/00z/ifs/0p25/oper/20240301000000-0h-oper-fc.grib2", "/huge/proc/weather-archive/ecmwf/20240301/00z/ifs/0p25/oper/20240301000000-0h-oper-fc.grib2")
# %%
grbs = pygrib.open("/huge/proc/weather-archive/ecmwf/20240301/00z/ifs/0p25/oper/20240301000000-0h-oper-fc.grib2")
levels = set()
for grb in grbs:
    print(grb)
    levels.add(grb.level)
len(levels)
# %%
hres_rt = np.load("/fast/proc/hres_rt/f000/2024030100.npz")
# %%
hres_rt["sfc"].shape, hres_rt["pr"].shape
# %%
from utils import levels_tiny
len(levels_tiny)
# %%
levels_tiny
# %%
levels














# %%
# pointy VS WM
# %%
with open("/fast/wbhaoxing/deep/evals/cities/city_newnewpointy_errors_summer.json", "r") as f:
    new_pointy_errors = json.load(f)
# compute RMSE for each city by forecast hour
new_pointy_rmse = defaultdict(list)
for city, errors in new_pointy_errors.items():
    for fh in list(range(24, 337, 24)):
        errors_fh = [error[4] for error in errors if error[1] == fh and not np.isnan(error[4])]
        rmse = np.sqrt(np.mean(np.square(errors_fh)))
        new_pointy_rmse[city].append(float(rmse))
new_pointy_rmse

with open("/fast/wbhaoxing/deep/evals/cities/city_oldpointy_errors_summer.json", "r") as f:
    old_pointy_errors = json.load(f)
# compute RMSE for each city by forecast hour
old_pointy_rmse = defaultdict(list)
for city, errors in old_pointy_errors.items():
    for fh in list(range(24, 337, 24)):
        errors_fh = [error[4] for error in errors if error[1] == fh and not np.isnan(error[4])]
        rmse = np.sqrt(np.mean(np.square(errors_fh)))
        old_pointy_rmse[city].append(float(rmse))
old_pointy_rmse
# %%
with open("/fast/wbhaoxing/deep/evals/cities/city_wm_rmse_big.json", "r") as f:
    city_rmse = json.load(f)


# %%
pointy_rmse = new_pointy_rmse

# %%
# make table of RMSE comparison
#row_labels = sorted(cities.keys())
row_labels = list(cities.keys())
pointy_rmses = np.array([pointy_rmse[cities[city].decode()] for city in row_labels])
wm_rmses = np.array([city_rmse[cities[city].decode()] for city in row_labels])
col_labels = [1,3,5,7,10,14]
pointy_rmses = pointy_rmses[:,np.array([0,2,4,6,9,13])] * 9/5
wm_rmses = wm_rmses[:,np.array([0,2,4,6,9,13])] * 9/5

fig, ax = plt.subplots(figsize=(8, 14), dpi=300)

ax.set_xticks(np.arange(len(col_labels)))
ax.set_yticks(np.arange(len(row_labels)))
ax.set_xticklabels(col_labels)
ax.set_yticklabels(row_labels)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(len(row_labels)):
    for j in range(len(col_labels)):
        text = ax.text(j, i, f"{wm_rmses[i, j]:.1f} | {pointy_rmses[i, j]:.1f}", ha="center", va="center", color="black")

cax = ax.imshow(wm_rmses - pointy_rmses, cmap='RdBu', alpha=0.8, vmin=-2.5, vmax=2.5, aspect=0.5)
cbar = fig.colorbar(cax, orientation='horizontal', pad=0.1, fraction=0.02)
cbar.ax.set_xticks([])
cbar.set_label("interpolated WM better                                         pointy better")

ax.set_title("2m Temperature, July-August 2024 METAR\nNew pointy vs WM", fontweight="bold")
secondary_x = ax.secondary_xaxis('top')
secondary_x.set_xlabel("interpolated WM RMSE (F) | new pointy RMSE (F)")
secondary_x.set_xticks([])
ax.set_xlabel("Forecast Lead Time (Days)")

#plt.tight_layout()
plt.show()
# %%
print("New pointy")
print(f"pointy 1-day RMSE: {np.mean(pointy_rmses[:,0]):.2f}, WM 1-day RMSE: {np.mean(wm_rmses[:,0]):.2f}, diff: {np.mean(wm_rmses[:,0]) - np.mean(pointy_rmses[:,0]):.2f}")
print(f"pointy 3-day RMSE: {np.mean(pointy_rmses[:,1]):.2f}, WM 3-day RMSE: {np.mean(wm_rmses[:,1]):.2f}, diff: {np.mean(wm_rmses[:,1]) - np.mean(pointy_rmses[:,1]):.2f}")
print(f"pointy 5-day RMSE: {np.mean(pointy_rmses[:,2]):.2f}, WM 5-day RMSE: {np.mean(wm_rmses[:,2]):.2f}, diff: {np.mean(wm_rmses[:,2]) - np.mean(pointy_rmses[:,2]):.2f}")
print(f"pointy 7-day RMSE: {np.mean(pointy_rmses[:,3]):.2f}, WM 7-day RMSE: {np.mean(wm_rmses[:,3]):.2f}, diff: {np.mean(wm_rmses[:,3]) - np.mean(pointy_rmses[:,3]):.2f}")
print(f"pointy 10-day RMSE: {np.mean(pointy_rmses[:,4]):.2f}, WM 10-day RMSE: {np.mean(wm_rmses[:,4]):.2f}, diff: {np.mean(wm_rmses[:,4]) - np.mean(pointy_rmses[:,4]):.2f}")
print(f"pointy 14-day RMSE: {np.mean(pointy_rmses[:,5]):.2f}, WM 14-day RMSE: {np.mean(wm_rmses[:,5]):.2f}, diff: {np.mean(wm_rmses[:,5]) - np.mean(pointy_rmses[:,5]):.2f}")
# %%
row_labels = list(cities.keys())
new_pointy_rmses = np.array([new_pointy_rmse[cities[city].decode()] for city in row_labels])
old_pointy_rmses = np.array([old_pointy_rmse[cities[city].decode()] for city in row_labels])
col_labels = [1,3,5,7,10,14]
new_pointy_rmses = new_pointy_rmses[:,np.array([0,2,4,6,9,13])] * 9/5
old_pointy_rmses = old_pointy_rmses[:,np.array([0,2,4,6,9,13])] * 9/5

fig, ax = plt.subplots(figsize=(8, 14), dpi=300)

ax.set_xticks(np.arange(len(col_labels)))
ax.set_yticks(np.arange(len(row_labels)))
ax.set_xticklabels(col_labels)
ax.set_yticklabels(row_labels)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(len(row_labels)):
    for j in range(len(col_labels)):
        text = ax.text(j, i, f"{old_pointy_rmses[i, j]:.1f} | {new_pointy_rmses[i, j]:.1f}", ha="center", va="center", color="black")

cax = ax.imshow(old_pointy_rmses - new_pointy_rmses, cmap='RdBu', alpha=0.8, vmin=-1, vmax=1, aspect=0.5)
cbar = fig.colorbar(cax, orientation='horizontal', pad=0.1, fraction=0.02)
cbar.ax.set_xticks([])
cbar.set_label("old pointy better                                        new pointy better")

ax.set_title("2m Temperature, July-August 2024 METAR\nNew pointy vs WM", fontweight="bold")
secondary_x = ax.secondary_xaxis('top')
secondary_x.set_xlabel("old | new pointy RMSE (F)")
secondary_x.set_xticks([])
ax.set_xlabel("Forecast Lead Time (Days)")

#plt.tight_layout()
plt.show()
# %%
