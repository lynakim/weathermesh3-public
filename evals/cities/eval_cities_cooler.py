from collections import defaultdict
from tqdm import tqdm
import json
import os
import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys
sys.path.append('/fast/wbhaoxing/windborne')
sys.path.append('/fast/wbhaoxing/deep')
from meteo.tools.process_dataset import download_s3_file
from hres.model import HresModel
from hres.inference import load_model, get_point_data_nb
from datetime import datetime, timezone, timedelta

def get_closest_utc_whole_hour(timestamp: int) -> datetime:
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    if dt.minute < 30:
        return dt.replace(minute=0, second=0)
    else:
        return (dt + timedelta(hours=1)).replace(minute=0, second=0)

cities = {
    "Boston": b"BOS",
    "New York": b"LGA",
    "Philadelphia": b"PHL",
    "Atlanta": b"ATL",
    "Miami": b"MIA",
    "Chicago": b"ORD",
    "Detroit": b"DTW",
    "Houston": b"IAH",
    "Los Angeles": b"LAX",
    "Seattle": b"SEA",
    "Honolulu": b"PHNL",
    "San Jose": b"SJC",
    "Tokyo": b"RJTT",
    "Beijing": b"ZBAA",
    "Mumbai": b"VABB",
    "Barcelona": b"LEBL",
    "Sao Paolo": b"SBGR",
    "London": b"EGLL",
    "Seoul": b"RKSI",
    "Vienna": b"LOWW",
    "Moscow": b"UUEE",
    "Toronto": b"CYYZ",
    "Sydney": b"YSSY",
    "Cairo": b"HECA",
    "Mexico City": b"MMMX",
    "Johannesburg": b"FAOR",
    "Dubai": b"OMDB",
    "Istanbul": b"LTBA",
    "Guam": b"PGUM",
    "Cape Verde": b"GVAC",
    "Nairobi": b"HKJK",
}

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

stations = pickle.load(open("/fast/proc/metar/stations.pickle", "rb"))
points = [stations[city] if city in stations else None for city in cities.values()]
jp_points = [jp_cities[city] for city in jp_cities]

# gather city observations
print("Gathering city observations")
city_obs = defaultdict(list)

for day in tqdm(range(1, 32)):
    for month in [7, 8]:
        if month == 2 and day > 29:
            continue
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

pointy_ckpt_path = '/huge/deep/runs_hres/run_Oct28-coolerpacific_20241028-114757/model_step146000_loss0.229.pt'
new = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True, n_out=8, grid="_small").half().to('cuda')
print(f"Loading pointy model from {pointy_ckpt_path}")
load_model(pointy_ckpt_path, new)

# compute WeatherMesh errors on the city observations
print("Computing pointy errors")
city_errors = defaultdict(list)

for month in [7, 8]:
    for day in tqdm(range(1, 32)):
        if month == 2 and day > 29:
            continue
        init_datestr = f"2024{month:02d}{day:02d}00"
        for fh in range(24, 337, 24):
            print(f"processing {init_datestr} forecast hour {fh}")
            obs_date = datetime(2024, month, day, 0, tzinfo=timezone.utc) + timedelta(hours=fh)
            os.makedirs(f"/huge/deep/realtime/outputs/WeatherMesh-backtest/{init_datestr}/det", exist_ok=True)
            download_s3_file("wb-dlnwp", f"WeatherMesh-backtest/{init_datestr}/det/{fh}.vGM0.npy", f"/huge/deep/realtime/outputs/WeatherMesh-backtest/{init_datestr}/det/{fh}.vGM0.npy")
            wm_out = np.load(f"/huge/deep/realtime/outputs/WeatherMesh-backtest/{init_datestr}/det/{fh}.vGM0.npy")
            pointy_out = get_point_data_nb(new, points, wm_out, obs_date)
            jp_pointy_out = get_point_data_nb(new, jp_points, wm_out, obs_date)

            for entry in city_obs[obs_date.timestamp()]:
                city = entry[0]
                city_index = list(cities.values()).index(city)
                tempc_pointy = pointy_out["forecasts"][city_index][0]["temperature_2m"]
                tempk_pointy = tempc_pointy + 273.15
                lon, lat, tempf = entry[2], entry[3], entry[5]
                tempk = float((tempf - 32) * 5 / 9 + 273.15)
                city_errors[entry[0].decode()].append((init_datestr, fh, tempk, tempk_pointy, tempk - tempk_pointy))
        with open("/fast/wbhaoxing/deep/evals/cities/city_cooler_errors_summer.json", "w") as f:
            json.dump(city_errors, f)