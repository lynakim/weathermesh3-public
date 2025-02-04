# %%
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

meta = json.load(open("/fast/realtime/outputs/WeatherMesh/meta.Qfiz.json", "r"))
lons = np.array(meta["lons"])
lats = np.array(meta["lats"])
conus_bounds = [-133.83, -61.8, 21.1, 62.2]
conus_lat_idx = np.where((lats >= conus_bounds[2]) & (lats <= conus_bounds[3]))[0]
conus_lon_idx = np.where((lons >= conus_bounds[0]) & (lons <= conus_bounds[1]))[0]
conus_lats = lats[conus_lat_idx]
conus_lons = lons[conus_lon_idx]

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
finish = datetime(2022, 12, 15, 0, 0, tzinfo=timezone.utc)
current = init_date
mean_era5_temp = {}
while current <= finish:
    ts = int(current.timestamp())
    era5_npz = np.load(f"/fast/proc/era5/f000/202212/{ts}.npz")
    era5_sfc, era5_pr = unnorm_era5(era5_npz)
    era5_temp = era5_sfc[:,:,2] - 273.15
    era5_temp_conus = era5_temp[conus_lat_idx][:, conus_lon_idx]
    mean_temp = era5_temp_conus.mean()
    print(f"{current.strftime('%Y-%m-%d %H')}: {mean_temp}")
    mean_era5_temp[ts] = float(mean_temp)
    current += timedelta(hours=1)
    with open("/fast/wbhaoxing/deep/evals/dec2022/mean_era5_temp.json", "w") as f:
        json.dump(mean_era5_temp, f)