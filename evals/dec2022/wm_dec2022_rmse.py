from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('/fast/wbhaoxing/windborne')
from meteo.tools.process_dataset import download_s3_file
import torch
from tqdm import tqdm
np.set_printoptions(precision=4, suppress=True)

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from PIL import Image
import pickle
import pygrib
import requests
from scipy.interpolate import RegularGridInterpolator

from utils import levels_joank, levels_medium, levels_hres, levels_tiny, levels_full, CONSTS_PATH, core_pressure_vars, core_sfc_vars

meta_hash = "Qfiz"
meta = json.load(open("/fast/realtime/outputs/WeatherMesh/meta.Qfiz.json", "r"))
lons = np.array(meta["lons"])
lats = np.array(meta["lats"])
conus_bounds = [-133.83, -61.8, 21.1, 62.2]
conus_lat_idx = np.where((lats >= conus_bounds[2]) & (lats <= conus_bounds[3]))[0]
conus_lon_idx = np.where((lons >= conus_bounds[0]) & (lons <= conus_bounds[1]))[0]
conus_lats = lats[conus_lat_idx]
conus_lons = lons[conus_lon_idx]

def get_temp(date: datetime, fh: int) -> np.ndarray:
    if os.path.exists(f"/huge/users/haoxing/dec2022/{date.strftime('%Y%m%d')}_{fh}.npy"):
        return np.load(f"/huge/users/haoxing/dec2022/{date.strftime('%Y%m%d')}_{fh}.npy")
    else:
        temp = np.load(f"/huge/deep/realtime/outputs/WeatherMesh-backtest/{date.strftime('%Y%m%d%H')}/det/{fh}.{meta_hash}.npy")[:,:,-5] - 273.15
        np.save(f"/huge/users/haoxing/dec2022/{date.strftime('%Y%m%d')}_{fh}.npy", temp)
        return temp

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

init_date = datetime(2022, 12, 13, 0, 0, tzinfo=timezone.utc)
end_date = datetime(2022, 12, 16, 0, 0, tzinfo=timezone.utc)
forecast_zero = init_date
rmse_dict, bias_dict = defaultdict(list), defaultdict(list)
while forecast_zero <= end_date:
    for fh in range(337):
        temp = get_temp(forecast_zero, fh)
        forecast_date = forecast_zero + timedelta(hours=fh)
        ts = int(forecast_date.timestamp())
        era5_npz = np.load(f"/fast/proc/era5/f000/202212/{ts}.npz")
        era5_sfc, era5_pr = unnorm_era5(era5_npz)
        era5_temp = era5_sfc[:,:,2] - 273.15
        temp_cropped = temp[conus_lat_idx][:, conus_lon_idx]
        era5_temp_cropped = era5_temp[conus_lat_idx][:, conus_lon_idx]
        rmse = np.sqrt(np.mean((temp_cropped - era5_temp_cropped)**2))
        bias = np.mean(temp_cropped - era5_temp_cropped)
        rmse_dict[forecast_zero.strftime("%Y%m%d%H")].append(float(rmse))
        bias_dict[forecast_zero.strftime("%Y%m%d%H")].append(float(bias))
        print(f"forecast zero: {forecast_zero}, forecast hour: {fh}, RMSE: {rmse}, bias: {bias}")
        # save
        with open("/fast/wbhaoxing/deep/evals/dec2022/conus_rmse.json", "w") as f:
            json.dump(rmse_dict, f)
        with open("/fast/wbhaoxing/deep/evals/dec2022/conus_bias.json", "w") as f:
            json.dump(bias_dict, f)
    forecast_zero += timedelta(hours=24)