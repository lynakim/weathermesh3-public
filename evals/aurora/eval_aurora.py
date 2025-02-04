# %%
import numpy as np
import os
import sys
sys.path.append('/fast/haoxing/windborne')
sys.path.append('/fast/haoxing/deep')
from meteo.tools.process_dataset import download_s3_file
np.set_printoptions(precision=4, suppress=True)
from datetime import datetime, timezone, timedelta
import pickle
import pygrib
from einops import rearrange

from utils import levels_joank, levels_medium, levels_tiny, levels_full, CONSTS_PATH, core_pressure_vars, core_sfc_vars, levels_aurora, levels_hres

ordered_pr_vars = ["z", "t", "u", "v", "q"]
ordered_sfc_vars = ["10u", "10v", "2t", "msl"]
# %%

def unnorm_era5(era5: np.lib.npyio.NpzFile, levels_out: list[int], levels_in: list[int] = levels_medium):
    with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
        norm = pickle.load(f)
    wh_lev_in_medium = np.array([levels_in.index(l) for l in levels_out])
    era5_sfc, era5_pr = era5["sfc"].astype(np.float32), era5["pr"][:,:,:,wh_lev_in_medium].astype(np.float32)

    wh_lev = np.array([levels_full.index(l) for l in levels_out])
    for i, v in enumerate(core_pressure_vars):
        mean, std2 = norm[v]
        era5_pr[:,:,i] = era5_pr[:,:,i] * np.sqrt(std2)[wh_lev] + mean[wh_lev]

    for i, v in enumerate(core_sfc_vars):
        mean, std2 = norm[v]
        era5_sfc[:,:,i] = era5_sfc[:,:,i] * np.sqrt(std2) + mean
    return era5_sfc[:720], era5_pr[:720]

def format_aurora_output(aurora_out: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    sfc = aurora_out["sfc"].item()
    pr = aurora_out["atmos"].item()
    sfc_arr, pr_arr = [], []
    for var in ordered_sfc_vars:
        sfc_arr.append(sfc[var]) # shape (720, 1440)
    for var in ordered_pr_vars:
        pr_arr.append(pr[var]) # shape (13, 720, 1440)
    sfc_arr = np.stack(sfc_arr, axis=0)
    pr_arr = np.stack(pr_arr, axis=0)
    sfc_arr = rearrange(sfc_arr, "sfc lat lon -> lat lon sfc")
    pr_arr = rearrange(pr_arr, "pr lev lat lon -> lat lon pr lev")
    return pr_arr, sfc_arr


def rmse_aurora_era5(aurora: np.lib.npyio.NpzFile, era5: np.lib.npyio.NpzFile):
    era5_sfc, era5_pr = unnorm_era5(era5, levels_aurora)
    aurora_pr, aurora_sfc = format_aurora_output(aurora)
    lats = np.linspace(90, -90, 720)
    weights = np.cos(lats * np.pi/180)
    weights = np.stack([weights] * 1440, axis=1)
    rmse_pr = np.sqrt(np.average((era5_pr - aurora_pr) ** 2, axis=(0, 1), weights=weights))
    rmse_sfc = np.sqrt(np.average((era5_sfc - aurora_sfc) ** 2, axis=(0, 1), weights=weights))
    return rmse_pr, rmse_sfc

def rmse_aurora_hrest0(aurora: np.lib.npyio.NpzFile, hrest0: np.lib.npyio.NpzFile):
    hrest0_sfc, hrest0_pr = unnorm_era5(hrest0, levels_aurora, levels_hres)
    aurora_pr, aurora_sfc = format_aurora_output(aurora)
    lats = np.linspace(90, -90, 720)
    weights = np.cos(lats * np.pi/180)
    weights = np.stack([weights] * 1440, axis=1)
    rmse_pr = np.sqrt(np.average((hrest0_pr - aurora_pr) ** 2, axis=(0, 1), weights=weights))
    rmse_sfc = np.sqrt(np.average((hrest0_sfc - aurora_sfc) ** 2, axis=(0, 1), weights=weights))
    return rmse_pr, rmse_sfc

def get_aurora_rmse(start_date: datetime, end_date: datetime):
    current_date = start_date
    while current_date <= end_date:
        for h in list(range(24, 14*24+1, 24)):
            print(f"Working on {current_date.strftime('%Y%m%d')} forecast hour {h:03d}")
            # oh gosh all the files are off by 6 hours
            aurora_npz = np.load(f"/huge/users/haoxing/aurora/finetuned4/{current_date.strftime('%Y%m%d%H')}/{h-6}.npz", allow_pickle=True)
            forecast_date = current_date + timedelta(hours=h)
            ts = int(forecast_date.timestamp())
            era5_npz = np.load(f"/fast/proc/era5/f000/2024{forecast_date.month:02d}/{ts}.npz")
            #hrest0_npz = np.load(f"/fast/proc/hres_rt/f000/{forecast_date.strftime('%Y%m%d%H')}.npz")
            rmse_era5_pr, rmse_era5_sfc = rmse_aurora_era5(aurora_npz, era5_npz)
            #rmse_hrest0_pr, rmse_hrest0_sfc = rmse_aurora_hrest0(aurora_npz, hrest0_npz)
            for i in range(4):
                print(f"{core_sfc_vars[i]}: vs era5 {rmse_era5_sfc[i]}")#, vs hrest0 {rmse_hrest0_sfc[i]}")
            # z500
            print(f"z500: vs era5 {rmse_era5_pr[0, 7]}")#, vs hrest0 {rmse_hrest0_pr[0, 7]}")
            np.save(f"/huge/users/haoxing/aurora/rmses4/vs_era5/pr_{current_date.strftime('%Y%m%d')}_00_{h:03d}_rmse.npy", rmse_era5_pr)
            np.save(f"/huge/users/haoxing/aurora/rmses4/vs_era5/sfc_{current_date.strftime('%Y%m%d')}_00_{h:03d}_rmse.npy", rmse_era5_sfc)
            # np.save(f"/huge/users/haoxing/aurora/rmses4/vs_hrest0/pr_{current_date.strftime('%Y%m%d')}_00_{h:03d}_rmse.npy", rmse_hrest0_pr)
            # np.save(f"/huge/users/haoxing/aurora/rmses4/vs_hrest0/sfc_{current_date.strftime('%Y%m%d')}_00_{h:03d}_rmse.npy", rmse_hrest0_sfc)

        current_date += timedelta(days=1)











if __name__ == "__main__":
    try:
        start_date_str = sys.argv[1]
        end_date_str = sys.argv[2]
    except:
        print("usage: python3 evals/eval_aurora.py <start_date> <end_date>")
        print("e.g. python3 evals/eval_aurora.py 20240101 20240131")
        sys.exit(1)
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    start_date = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_date = datetime.strptime(end_date_str, "%Y%m%d")
    end_date = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc)
    get_aurora_rmse(start_date, end_date)