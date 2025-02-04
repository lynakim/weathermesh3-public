from collections import defaultdict
import json
import numpy as np
import os
import sys
sys.path.append('/fast/wbhaoxing/windborne')
sys.path.append('/fast/wbhaoxing/deep')
from meteo.tools.process_dataset import download_s3_file
np.set_printoptions(precision=4, suppress=True)
from datetime import datetime, timezone, timedelta
import pickle
import pygrib

from utils import levels_joank, levels_medium, levels_tiny, levels_full, CONSTS_PATH, core_pressure_vars, core_sfc_vars

# %%
ifs_data_path = '/huge/proc/weather-archive/ecmwf/'
s3_bucket = 'wb-weather-archive'

VAR_NAMES = {
    "Geopotential Height": "129_z",
    "Temperature": "130_t",
    "U component of wind": "131_u",
    "V component of wind": "132_v",
    "Specific humidity": "133_q"
}

SURFACE_VARS = {
    "10 metre U wind component": "165_10u",
    "10 metre V wind component": "166_10v",
    "2 metre temperature": "167_2t",
    "Mean sea level pressure": "151_msl"
}

meta_hash = "Qfiz"
meta = json.load(open("/fast/realtime/outputs/WeatherMesh/meta.Qfiz.json", "r"))
lons = np.array(meta["lons"])
lats = np.array(meta["lats"])
conus_bounds = [-133.83, -61.8, 21.1, 62.2]
conus_lat_idx = np.where((lats >= conus_bounds[2]) & (lats <= conus_bounds[3]))[0]
conus_lon_idx = np.where((lons >= conus_bounds[0]) & (lons <= conus_bounds[1]))[0]
conus_lats = lats[conus_lat_idx]
conus_lons = lons[conus_lon_idx]

def get_temp(date: datetime, fh: int, model: str) -> np.ndarray:
    assert model in ["wm", "era5", "hres"]
    if model == "wm":
        if os.path.exists(f"/huge/users/haoxing/dec2022/{date.strftime('%Y%m%d')}_{fh}.npy"):
            return np.load(f"/huge/users/haoxing/dec2022/{date.strftime('%Y%m%d')}_{fh}.npy")
        else:
            temp = np.load(f"/huge/deep/realtime/outputs/WeatherMesh-backtest/{date.strftime('%Y%m%d%H')}/det/{fh}.{meta_hash}.npy")[:,:,-5] - 273.15
            np.save(f"/huge/users/haoxing/dec2022/{date.strftime('%Y%m%d')}_{fh}.npy", temp)
            return temp
    elif model == "era5":
        valid_date = date + timedelta(hours=fh)
        ts = int(valid_date.timestamp())
        era5_npz = np.load(f"/fast/proc/era5/f000/{valid_date.strftime('%Y%m')}/{ts}.npz")
        era5_sfc, era5_pr = unnorm_era5(era5_npz)
        era5_temp = era5_sfc[:,:,2] - 273.15
        return era5_temp
    elif model == "hres":
        return np.load(f"/huge/users/haoxing/ifs/temp2m_{date.strftime('%Y%m%d%H')}_{fh}.npy")[:720] - 273.15

def get_heating_degree_days(mint, maxt):
    return 65 - ((mint + maxt) / 2 * 9 / 5 + 32)

def get_minmax_temp(forecast_zero: datetime, date: datetime, model: str) -> np.ndarray:
    """returns the mininum and maximum of all hourly temperature forecasts between
       date and date + 24 hours"""
    if model == "hres":
        # for hres we have min/max over the past 6h
        min_temps, max_temps = [], []
        fh_min = int((date - forecast_zero).total_seconds() // 3600)
        fh_min_c = fh_min // 6 * 6
        assert fh_min_c == fh_min
        fhs = [fh_min_c + i * 6 for i in range(1,5)]
        fzstr = forecast_zero.strftime("%Y%m%d%H")
        print(f"forecast zero: {forecast_zero}, forecast hour {fh_min}")
        print(f"fhs: {fhs}")
        for fh in fhs:
            min_temps.append(np.load(f"/huge/users/haoxing/ifs/min2t6_{fzstr}_{fh}.npy")[:720] - 273.15)
            max_temps.append(np.load(f"/huge/users/haoxing/ifs/max2t6_{fzstr}_{fh}.npy")[:720] - 273.15)
        min_temps = np.stack(min_temps, axis=0)
        max_temps = np.stack(max_temps, axis=0)
        mint, maxt = min_temps.min(axis=0), max_temps.max(axis=0)
        hdd = get_heating_degree_days(mint, maxt)
        return mint, maxt, hdd
    else:
        temps = []
        fh_min = int((date - forecast_zero).total_seconds() // 3600)
        print(f"forecast zero: {forecast_zero}, forecast hour {fh_min}")
        for fh in range(fh_min, fh_min+24):
            try:
                temp = get_temp(forecast_zero, fh, model)
            except:
                pass
            temps.append(temp)
        temps = np.stack(temps, axis=0)
        return temps.min(axis=0), temps.max(axis=0)

def unnorm_era5(era5: np.lib.npyio.NpzFile, levels: list[int]):
    with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
        norm = pickle.load(f)
    wh_lev_in_medium = np.array([levels_medium.index(l) for l in levels])
    era5_sfc, era5_pr = era5["sfc"].astype(np.float32), era5["pr"][:,:,:,wh_lev_in_medium].astype(np.float32)

    wh_lev = np.array([levels_full.index(l) for l in levels])
    for i, v in enumerate(core_pressure_vars):
        mean, std2 = norm[v]
        era5_pr[:,:,i] = era5_pr[:,:,i] * np.sqrt(std2)[wh_lev] + mean[wh_lev]

    for i, v in enumerate(core_sfc_vars):
        mean, std2 = norm[v]
        era5_sfc[:,:,i] = era5_sfc[:,:,i] * np.sqrt(std2) + mean
    return era5_sfc, era5_pr

def get_hres_rmse(start_date: datetime, end_date: datetime):
    current_date = start_date
    min_rmse_dict, min_bias_dict = defaultdict(list), defaultdict(list)
    max_rmse_dict, max_bias_dict = defaultdict(list), defaultdict(list)
    hdd_rmse_dict, hdd_bias_dict, hdd_mean_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    while current_date <= end_date:
        era5_mins = np.load(f"/huge/users/haoxing/dec2022/min_temps/era5_min_{current_date.strftime('%Y%m%d')}.npy")
        era5_maxs = np.load(f"/huge/users/haoxing/dec2022/max_temps/era5_max_{current_date.strftime('%Y%m%d')}.npy")
        for i in range(10):
            print(f"Working on forecast zero {current_date.strftime('%Y%m%d')}, date {current_date + timedelta(days=i)}")
            valid_date = current_date + timedelta(days=i)
            hres_min, hres_max, hres_hdd = get_minmax_temp(current_date, valid_date, "hres")
            era5_min = era5_mins[i-1]
            era5_max = era5_maxs[i-1]
            hres_min_conus = hres_min[conus_lat_idx][:, conus_lon_idx]
            hres_max_conus = hres_max[conus_lat_idx][:, conus_lon_idx]
            hres_hdd_conus = hres_hdd[conus_lat_idx][:, conus_lon_idx]
            era5_min_conus = era5_min[conus_lat_idx][:, conus_lon_idx]
            era5_max_conus = era5_max[conus_lat_idx][:, conus_lon_idx]
            era5_hdd_conus = get_heating_degree_days(era5_min_conus, era5_max_conus)
            np.save(f'/fast/wbhaoxing/deep/evals/dec2022/hdd/era5_{current_date.strftime("%Y%m%d")}_{valid_date.strftime("%Y%m%d")}.npy', era5_hdd_conus)
            rmse_min = np.sqrt(np.mean((era5_min_conus - hres_min_conus)**2))
            rmse_max = np.sqrt(np.mean((era5_max_conus - hres_max_conus)**2))
            rmse_hdd = np.sqrt(np.mean((era5_hdd_conus - hres_hdd_conus)**2))
            np.save(f'/fast/wbhaoxing/deep/evals/dec2022/hdd/hres_{current_date.strftime("%Y%m%d")}_{valid_date.strftime("%Y%m%d")}.npy', hres_hdd_conus)
            bias_min = np.mean(hres_min_conus - era5_min_conus)
            bias_max = np.mean(hres_max_conus - era5_max_conus)
            bias_hdd = np.mean(hres_hdd_conus - era5_hdd_conus)
            print(f"RMSE min: {rmse_min:.3f} Bias min: {bias_min:.3f}")
            print(f"RMSE max: {rmse_max:.3f} Bias max: {bias_max:.3f}")
            print(f"RMSE hdd: {rmse_hdd:.3f} Bias hdd: {bias_hdd:.3f}")
            min_rmse_dict[current_date.strftime("%Y%m%d")].append(float(rmse_min))
            min_bias_dict[current_date.strftime("%Y%m%d")].append(float(bias_min))
            max_rmse_dict[current_date.strftime("%Y%m%d")].append(float(rmse_max))
            max_bias_dict[current_date.strftime("%Y%m%d")].append(float(bias_max))
            hdd_rmse_dict[current_date.strftime("%Y%m%d")].append(float(rmse_hdd))
            hdd_bias_dict[current_date.strftime("%Y%m%d")].append(float(bias_hdd))
            hdd_mean_dict[current_date.strftime("%Y%m%d")].append(float(np.mean(hres_hdd_conus)))
            with open("/fast/wbhaoxing/deep/evals/dec2022/hres_mint_rmses.json", "w") as f:
                json.dump(min_rmse_dict, f)
            with open("/fast/wbhaoxing/deep/evals/dec2022/hres_mint_biases.json", "w") as f:
                json.dump(min_bias_dict, f)
            with open("/fast/wbhaoxing/deep/evals/dec2022/hres_maxt_rmses.json", "w") as f:
                json.dump(max_rmse_dict, f)
            with open("/fast/wbhaoxing/deep/evals/dec2022/hres_maxt_biases.json", "w") as f:
                json.dump(max_bias_dict, f)
            with open("/fast/wbhaoxing/deep/evals/dec2022/hres_hdd_rmses.json", "w") as f:
                json.dump(hdd_rmse_dict, f)
            with open("/fast/wbhaoxing/deep/evals/dec2022/hres_hdd_biases.json", "w") as f:
                json.dump(hdd_bias_dict, f)
            with open("/fast/wbhaoxing/deep/evals/dec2022/hres_hdd_means.json", "w") as f:
                json.dump(hdd_mean_dict, f)

        current_date += timedelta(days=1)

if __name__ == "__main__":
    try:
        start_date_str = sys.argv[1]
        end_date_str = sys.argv[2]
    except:
        print("usage: python3 evals/eval_hres.py <start_date> <end_date>")
        print("e.g. python3 evals/eval_hres.py 20240101 20240131")
        sys.exit(1)
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    start_date = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_date = datetime.strptime(end_date_str, "%Y%m%d")
    end_date = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc)
    get_hres_rmse(start_date, end_date)