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
from utils import CONSTS_PATH, levels_joank, levels_medium, levels_full, core_pressure_vars, core_sfc_vars

meta = json.load(open("/huge/deep/realtime/outputs/yamahabachelor/meta.vGM0.json", "r"))
wm_lats, wm_lons = meta["lats"], meta["lons"]
wm_lons = wm_lons[720:] + wm_lons[:720]

def get_closest_utc_whole_hour(timestamp: int) -> datetime:
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    if dt.minute < 30:
        return dt.replace(minute=0, second=0)
    else:
        return (dt + timedelta(hours=1)).replace(minute=0, second=0)

def unnorm_era5(era5: np.lib.npyio.NpzFile):
    with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
        norm = pickle.load(f)
    wh_lev_joank_medium = np.array([levels_medium.index(l) for l in levels_joank])
    era5_sfc, era5_pr = era5["sfc"].astype(np.float32), era5["pr"][:,:,:,wh_lev_joank_medium].astype(np.float32)

    wh_lev = np.array([levels_full.index(l) for l in levels_joank])
    for i, v in enumerate(core_pressure_vars):
        mean, std2 = norm[v]
        era5_pr[:,:,i] = era5_pr[:,:,i] * np.sqrt(std2)[wh_lev] + mean[wh_lev]

    for i, v in enumerate(core_sfc_vars):
        mean, std2 = norm[v]
        era5_sfc[:,:,i] = era5_sfc[:,:,i] * np.sqrt(std2) + mean
    return era5_sfc, era5_pr

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
jp_points = [jp_cities[city] for city in jp_cities]


pointy_ckpt_path = '/huge/deep/runs_hres/run_Oct28-coolerpacific_20241028-114757/model_step146000_loss0.229.pt'
new = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True, n_out=8, grid="_small").half().to('cuda')
print(f"Loading pointy model from {pointy_ckpt_path}")
load_model(pointy_ckpt_path, new)

# compute WeatherMesh errors on the city observations
print("Computing pointy errors")
city_errors = defaultdict(list)

for month in [1, 2]:
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
            jp_pointy_out = get_point_data_nb(new, jp_points, wm_out, obs_date)
            # interpolate era5
            era5 = np.load(f"/fast/proc/era5/f000/2024{obs_date.month:02d}/{int(obs_date.timestamp())}.npz")
            era5_sfc, era_pr = unnorm_era5(era5)
            era5_temp = era5_sfc[:,:,2] - 273.15
            era5_temp = np.concatenate([era5_temp[:,720:], era5_temp[:,:720]], axis=1)
            era5_interp = RegularGridInterpolator((wm_lats, wm_lons), era5_temp, method="linear")
            wm_temp = wm_out[:,:,-5] - 273.15
            wm_temp = np.concatenate([wm_temp[:,720:], wm_temp[:,:720]], axis=1)
            wm_interp = RegularGridInterpolator((wm_lats, wm_lons), wm_temp, method="linear")
            if fh <= 240:
                # hres_npz = np.load(f"/huge/users/haoxing/ifs/{init_datestr[:8]}_00z_{fh}h.npz")
                # hres_pr, hres_sfc = hres_npz["pr"][:720], hres_npz["sfc"][:720]
                # hres_sfc = np.concatenate([hres_sfc[:, 720:], hres_sfc[:, :720]], axis=1)
                # hres_temp = hres_sfc[:,:,2] - 273.15
                hres_temp = np.load(f"/huge/users/haoxing/ifs/temp2m_{init_datestr}_{fh}.npy")[:720] - 273.15
                hres_temp = np.concatenate([hres_temp[:, 720:], hres_temp[:, :720]], axis=1)
                hres_interp = RegularGridInterpolator((wm_lats, wm_lons), hres_temp, method="linear")
            for city, (lat, lon) in jp_cities.items():
                temp_era5 = era5_interp([lat, lon])[0]
                temp_wm = wm_interp([lat, lon])[0]
                temp_hres = hres_interp([lat, lon])[0] if fh <= 240 else np.nan
                temp_pointy = jp_pointy_out["forecasts"][list(jp_cities.keys()).index(city)][0]["temperature_2m"]
                print(f"{city}: era5: {temp_era5:.3f} wm: {temp_wm:.3f} hres: {temp_hres:.3f} pointy: {temp_pointy}")
                city_errors[city].append((init_datestr, fh, temp_pointy, temp_era5, temp_wm, temp_hres))

        with open("/fast/wbhaoxing/deep/evals/cities/jp_pointy_era5_wm_hres.json", "w") as f:
            json.dump(city_errors, f)