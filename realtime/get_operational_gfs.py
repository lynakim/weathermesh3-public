import os
import sys

import requests
import traceback
from datetime import datetime, timedelta
import string
import random
import numpy as np
import pygrib
import pickle
import builtins
from .data_fetch_helpers import run_from_argv


from utils import *
globals().update(NeoDatasetConfig().__dict__)


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

norm = None


def process(grb_files, output_path):
    global norm
    if norm is None:
        with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
            norm = pickle.load(f)

    prdic = {v: [] for k, v in VAR_NAMES.items()}
    whlev = {v: [] for k, v in VAR_NAMES.items()}
    sfcdic = {v: [] for k, v in SURFACE_VARS.items()}

    for grb_file in grb_files:
        grbs = pygrib.open(grb_file)
        for grb in grbs:
            name = grb.name
            name = name.replace("height", "Height")

            if name in VAR_NAMES and grb.typeOfLevel == "isobaricInhPa" and grb.level in levels_gfs:
                scaling = 1
                if name == "Geopotential Height":
                    scaling = 9.80665

                prdic[VAR_NAMES[name]].append(grb.values[:].copy().astype(np.float32) * scaling)
                whlev[VAR_NAMES[name]].append(int(grb.level))

            if name in SURFACE_VARS:
                sfcdic[SURFACE_VARS[name]].append(grb.values[:].copy().astype(np.float32))

    pr = []
    sfc = []
    for v in pressure_vars:
        mean, std2 = norm[v]
        if whlev[v] != levels_gfs:
            print("uhh")
            for a, b in zip(whlev[v], levels_gfs):
                print(a, b)
        assert whlev[v] == levels_gfs
        wh_lev = [levels_full.index(l) for l in levels_gfs]
        x = np.array(prdic[v]).transpose(1, 2, 0)
        x = (x - mean[np.newaxis, np.newaxis, wh_lev]) / np.sqrt(std2)[np.newaxis, np.newaxis, wh_lev]
        print(v, np.mean(x), np.std(x), np.max(np.abs(x)))
        pr.append(x.astype(np.float16))

    for v in sfc_vars:
        mean, std2 = norm[v]
        x = np.array(sfcdic[v])[0]
        x = (x - mean[np.newaxis, np.newaxis]) / np.sqrt(std2)[np.newaxis, np.newaxis]
        print(f"{v}: mean={round(np.mean(x), 3)} std={round(np.std(x), 3)}")
        sfc.append(x[0].astype(np.float16))

    print("Surface vars shape", np.array(sfc).shape)
    print("Pressure vars shape", np.array(pr).shape)

    sfc_data = np.array(sfc).transpose(1, 2, 0)
    data = np.array(pr).transpose(1, 2, 0, 3)
    pathp = output_path.replace(".npz", ".tmp.npz")
    np.savez(pathp, pr=data, sfc=sfc_data)
    os.rename(pathp, output_path)


def download_and_process(cycle_time, forecast_hour=0):
    print(f"Attempting to get GFS at {cycle_time}")

    output_path = "/fast/proc/gfs_rt/f%03d/%d.npz" % (forecast_hour, int(cycle_time.timestamp()))
    if os.path.exists(output_path):
        print(f"GFS at {cycle_time} has already been downloaded ({output_path})")
        return True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.%04d%02d%02d/%02d/atmos/gfs.t%02dz.pgrb2.0p25.f%03d" % (cycle_time.year, cycle_time.month, cycle_time.day, cycle_time.hour, cycle_time.hour, forecast_hour)
    urls = [base_url, base_url.replace('.pgrb2.', '.pgrb2b.')]

    tmp_files = []

    try:
        for url in urls:
            r = requests.get(url)

            if r.status_code == 404:
                print(f"File {url} not found")
                return False

            print(f"Downloading {url}", flush=True)
            tmp_file = "/tmp/" + ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
            with open(tmp_file, 'wb') as f:
                f.write(r.content)

            tmp_files.append(tmp_file)

        process(tmp_files, output_path)

        for tmp_file in tmp_files:
            os.remove(tmp_file)

        return True
    finally:
        try:
            for tmp_file in tmp_files:
                try:
                    os.remove(tmp_file)
                except:
                    pass
        except:
            print("Cleanup failed")
            traceback.print_exc()
            pass


def main():
    run_from_argv(download_and_process, 'gfs_rt')


if __name__ == '__main__':
    main()
