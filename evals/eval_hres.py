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

def process_hres(grb_files: list[str], output_path: str):
    """Like the process function in realtime/get_operational_gfs.py, but
       for HRES. It also does not normalize the data and uses float32."""
    
    prdic = {v: [] for k, v in VAR_NAMES.items()}
    whlev = {v: [] for k, v in VAR_NAMES.items()}
    sfcdic = {v: [] for k, v in SURFACE_VARS.items()}

    for grb_file in grb_files:
        grbs = pygrib.open(grb_file)
        for grb in grbs:
            name = grb.name
            name = name.replace("height", "Height")

            if name in VAR_NAMES and grb.typeOfLevel == "isobaricInhPa" and grb.level in levels_tiny:
                scaling = 1
                if name == "Geopotential Height":
                    scaling = 9.80665

                prdic[VAR_NAMES[name]].append((grb.level, grb.values[:].copy().astype(np.float32) * scaling))
                whlev[VAR_NAMES[name]].append(int(grb.level))

            if name in SURFACE_VARS:
                sfcdic[SURFACE_VARS[name]].append(grb.values[:].copy().astype(np.float32))

    pr = []
    sfc = []
    for v in core_pressure_vars:
        sorted_levels = sorted(whlev[v])
        # sort data by level
        prdic[v].sort(key=lambda x: x[0])

        if sorted_levels != levels_tiny:
            print("uhh")
            for a, b in zip(sorted_levels, levels_tiny):
                print(a, b)
        assert sorted_levels == levels_tiny
        values = [x[1] for x in prdic[v]]
        x = np.stack(values).transpose(1, 2, 0)
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

def rmse_hres_era5(hres: np.lib.npyio.NpzFile, era5: np.lib.npyio.NpzFile):
    era5_sfc, era5_pr = unnorm_era5(era5, levels_tiny)
    hres_pr, hres_sfc = hres["pr"][:720], hres["sfc"][:720]
    # change to 0 longitude
    hres_pr = np.concatenate([hres_pr[:, 720:], hres_pr[:, :720]], axis=1)
    hres_sfc = np.concatenate([hres_sfc[:, 720:], hres_sfc[:, :720]], axis=1)
    lats = np.linspace(90, -90, 720)
    weights = np.cos(lats * np.pi/180)
    weights = np.stack([weights] * 1440, axis=1)
    rmse_pr = np.sqrt(np.average((era5_pr - hres_pr) ** 2, axis=(0, 1), weights=weights))
    rmse_sfc = np.sqrt(np.average((era5_sfc - hres_sfc) ** 2, axis=(0, 1), weights=weights))
    return rmse_pr, rmse_sfc

def download_and_process_hres(init_date_str: str, forecast_hour: int):
    # s3://wb-weather-archive/ecmwf/20240319/00z/ifs/0p25/oper/20240319000000-24h-oper-fc.grib2
    s3_key = f"ecmwf/{init_date_str}/00z/ifs/0p25/oper/{init_date_str}000000-{forecast_hour}h-oper-fc.grib2"
    local_path = f"{ifs_data_path}/{init_date_str}/00z/ifs/0p25/oper/{init_date_str}000000-{forecast_hour}h-oper-fc.grib2"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    #download_s3_file(s3_bucket, s3_key, local_path)
    process_hres([local_path], f"/huge/users/haoxing/ifs/{init_date_str}_00z_{forecast_hour}h.npz")

def get_hres_rmse(start_date: datetime, end_date: datetime):
    current_date = start_date
    while current_date <= end_date:
        #for h in list(range(24, 10*24, 24)) + [234]:
        for h in list(range(24, 10*24+1, 24)):
            print(f"Working on {current_date.strftime('%Y%m%d')} forecast hour {h:03d}")
            download_and_process_hres(current_date.strftime("%Y%m%d"), h)
            hres_npz = np.load(f"/huge/users/haoxing/ifs/{current_date.strftime('%Y%m%d')}_00z_{h}h.npz")
            forecast_date = current_date + timedelta(hours=h)
            ts = int(forecast_date.timestamp())
            era5_npz = np.load(f"/fast/proc/era5/f000/2024{forecast_date.month:02d}/{ts}.npz")
            rmse_pr, rmse_sfc = rmse_hres_era5(hres_npz, era5_npz)
            for i in range(4):
                print(f"{core_sfc_vars[i]}: {rmse_sfc[i]}")
            np.save(f"/huge/users/haoxing/ifs/rmses/pr_{current_date.strftime('%Y%m%d')}_00_{h:03d}_rmse.npy", rmse_pr)
            np.save(f"/huge/users/haoxing/ifs/rmses/sfc_{current_date.strftime('%Y%m%d')}_00_{h:03d}_rmse.npy", rmse_sfc)

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