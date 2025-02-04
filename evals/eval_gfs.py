import numpy as np
import os
import sys
sys.path.append('/fast/wbhaoxing/windborne')
from meteo.tools.process_dataset import download_s3_file
np.set_printoptions(precision=4, suppress=True)
from datetime import datetime, timezone, timedelta
import pickle
import pygrib

from utils import levels_joank, levels_medium, levels_full, CONSTS_PATH, core_pressure_vars, core_sfc_vars

gfs_bucket = "noaa-gfs-bdp-pds"
gfs_data_path = "/huge/users/haoxing/gfs"

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
    "Pressure reduced to MSL": "151_msl"
}

def process(grb_files, output_path):
    """Like the process function is realtime/get_operational_gfs.py, but
       1) unnormalized
       2) for levels_joank instead of levels_gfs """
    
    prdic = {v: [] for k, v in VAR_NAMES.items()}
    whlev = {v: [] for k, v in VAR_NAMES.items()}
    sfcdic = {v: [] for k, v in SURFACE_VARS.items()}

    for grb_file in grb_files:
        grbs = pygrib.open(grb_file)
        for grb in grbs:
            name = grb.name
            name = name.replace("height", "Height")

            if name in VAR_NAMES and grb.typeOfLevel == "isobaricInhPa" and grb.level in levels_joank:
                scaling = 1
                if name == "Geopotential Height":
                    scaling = 9.80665

                prdic[VAR_NAMES[name]].append(grb.values[:].copy().astype(np.float32) * scaling)
                whlev[VAR_NAMES[name]].append(int(grb.level))

            if name in SURFACE_VARS:
                sfcdic[SURFACE_VARS[name]].append(grb.values[:].copy().astype(np.float32))

    pr = []
    sfc = []
    for v in core_pressure_vars:
        if whlev[v] != levels_joank:
            print("uhh")
            for a, b in zip(whlev[v], levels_joank):
                print(a, b)
        assert whlev[v] == levels_joank
        x = np.array(prdic[v]).transpose(1, 2, 0)
        print(v, np.mean(x), np.std(x), np.max(np.abs(x)))
        pr.append(x)

    for v in core_sfc_vars:
        x = np.array(sfcdic[v])
        print(f"{v}: mean={round(np.mean(x), 3)} std={round(np.std(x), 3)}")
        sfc.append(x[0])

    sfc_data = np.array(sfc).transpose(1, 2, 0)
    data = np.array(pr).transpose(1, 2, 0, 3)
    pathp = output_path.replace(".npz", ".tmp.npz")
    np.savez(pathp, pr=data, sfc=sfc_data)
    os.rename(pathp, output_path)

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

def rmse_gfs_era5(gfs: np.lib.npyio.NpzFile, era5: np.lib.npyio.NpzFile):
    era5_sfc, era5_pr = unnorm_era5(era5)
    gfs_pr, gfs_sfc = gfs["pr"][:720], gfs["sfc"][:720]
    lats = np.linspace(90, -90, 720)
    weights = np.cos(lats * np.pi/180)
    weights = np.stack([weights] * 1440, axis=1)
    rmse_pr = np.sqrt(np.average((era5_pr - gfs_pr) ** 2, axis=(0, 1), weights=weights))
    rmse_sfc = np.sqrt(np.average((era5_sfc - gfs_sfc) ** 2, axis=(0, 1), weights=weights))
    return rmse_pr, rmse_sfc

def download_and_process_gfs(init_date_str: str, forecast_hour: int):
    s3_key = f"gfs.{init_date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f{forecast_hour:03d}"
    local_path = f"{gfs_data_path}/gfs_{init_date_str}_00_{forecast_hour:03d}.grib2"
    download_s3_file(gfs_bucket, s3_key, local_path)
    process([local_path], f"{gfs_data_path}/gfs_{init_date_str}_00_{forecast_hour:03d}.npz")

def get_gfs_rmse(start_date: datetime, end_date: datetime):
    current_date = start_date
    while current_date <= end_date:
        for h in range(24, 14*24+1, 24):
            print(f"Working on {current_date.strftime('%Y%m%d')} forecast hour {h:03d}")
            download_and_process_gfs(current_date.strftime("%Y%m%d"), h)
            gfs_npz = np.load(f"{gfs_data_path}/gfs_{current_date.strftime('%Y%m%d')}_00_{h:03d}.npz")
            forecast_date = current_date + timedelta(hours=h)
            ts = int(forecast_date.timestamp())
            era5_npz = np.load(f"/fast/proc/era5/f000/2024{forecast_date.month:02d}/{ts}.npz")
            rmse_pr, rmse_sfc = rmse_gfs_era5(gfs_npz, era5_npz)
            for i in range(4):
                print(f"{core_sfc_vars[i]}: {rmse_sfc[i]}")
            np.save(f"{gfs_data_path}/rmses/pr_{current_date.strftime('%Y%m%d')}_00_{h:03d}_rmse.npy", rmse_pr)
            np.save(f"{gfs_data_path}/rmses/sfc_{current_date.strftime('%Y%m%d')}_00_{h:03d}_rmse.npy", rmse_sfc)

        current_date += timedelta(days=1)

if __name__ == "__main__":
    try:
        start_date_str = sys.argv[1]
        end_date_str = sys.argv[2]
    except:
        print("usage: python3 -m evals.eval_gfs <start_date> <end_date>")
        print("e.g. python3 -m evals.eval_gfs 20240101 20240131")
        sys.exit(1)
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    start_date = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_date = datetime.strptime(end_date_str, "%Y%m%d")
    end_date = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc)
    get_gfs_rmse(start_date, end_date)