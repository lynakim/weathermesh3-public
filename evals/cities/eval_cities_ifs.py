from collections import defaultdict
from tqdm import tqdm
import json
import os
import pygrib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys
sys.path.append('/fast/wbhaoxing/windborne')
from meteo.tools.process_dataset import download_s3_file
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
    "Porto Alegre": b"SBPA",
    "Belem": b"SBBE",
}

meta = json.load(open("/huge/deep/realtime/outputs/yamahabachelor/meta.vGM0.json", "r"))
wm_lats, wm_lons = meta["lats"], meta["lons"]
wm_lons = wm_lons[720:] + wm_lons[:720]

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

# compute IFS errors on the city observations
print("Computing IFS errors")
city_hres_errors = defaultdict(list)

for month in [7, 8]:
    for day in tqdm(range(1, 32)):
        if month == 2 and day > 29:
            continue
        init_datestr = f"2024{month:02d}{day:02d}00"
        for fh in list(range(24, 10*24+1, 24)):
            print(f"processing {init_datestr} forecast hour {fh}")
            obs_date = datetime(2024, month, day, 0, tzinfo=timezone.utc) + timedelta(hours=fh)
            # os.makedirs(f"/huge/proc/weather-archive/ecmwf/{init_datestr[:-2]}/00z/ifs/0p25/oper", exist_ok=True)
            # download_s3_file("wb-weather-archive", f"ecmwf/{init_datestr[:-2]}/00z/ifs/0p25/oper/{init_datestr}0000-{fh}h-oper-fc.grib2", f"/huge/proc/weather-archive/ecmwf/{init_datestr[:-2]}/00z/ifs/0p25/oper/{init_datestr}0000-{fh}h-oper-fc.grib2")
            grbs = pygrib.open(f"/huge/proc/weather-archive/ecmwf/{init_datestr[:-2]}/00z/ifs/0p25/oper/{init_datestr}0000-{fh}h-oper-fc.grib2")
            hres_out = grbs.select(name="2 metre temperature")[0].values[:720,:]
            #hres_out = np.load(f"/huge/users/haoxing/ifs/temp2m_{init_datestr}_{fh}.npy")[:720]
            tempk_interp = RegularGridInterpolator((wm_lats, wm_lons), hres_out, method="linear")
            for entry in city_obs[obs_date.timestamp()]:
                lon, lat, tempf = entry[2], entry[3], entry[5]
                tempk = (tempf - 32) * 5 / 9 + 273.15
                tempk_hres = tempk_interp([lat, lon])[0]
                city_hres_errors[entry[0].decode()].append((init_datestr, fh, float(tempk), float(tempk_hres), float(tempk - tempk_hres)))
        with open("/fast/haoxing/deep/evals/city_hres_errors_brazil_summer.json", "w") as f:
            json.dump(city_hres_errors, f)