from datetime import datetime, timedelta, timezone
import os
import pygrib
import sys
import time
import requests
from requests.auth import HTTPBasicAuth
import boto3
import getpass
from .data_fetch_helpers import run_from_argv

from utils import *

globals().update(DEFAULT_CONFIG.__dict__)

grid = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
norms, _ = load_state_norm(grid.wh_lev, DEFAULT_CONFIG)

wh_lev = [levels_full.index(l) for l in levels_hres]

S3_CLIENT = boto3.client('s3', region_name='us-west-2')

pvarlist = ["Geopotential", "Temperature", "U component of wind", "V component of wind", "Specific humidity"]
svarlist = ["10 metre U wind component", "10 metre V wind component", "2 metre temperature", "Mean sea level pressure"]

varlist = pvarlist + svarlist

EXPORTED_SURFACE_VARS = svarlist
EXPORTED_UPPER_LEVEL_VARS = ["Geopotential", "Temperature", "U component of wind", "V component of wind"]
EXPORTED_LEVELS = [500, 850]
PARTIAL_EXPORT_DIR = "/viz/data/hres"
EXPORTING = os.path.exists(PARTIAL_EXPORT_DIR) and getpass.getuser() != "windborne"

GRB_NAMES = {
    "Temperature": "t",
    "Geopotential": "geopotential",
    "U component of wind": "u",
    "V component of wind": "v",
    "Mean sea level pressure": "mslp",
    "2 metre temperature": "t2m",
    "10 metre U wind component": "u10m",
    "10 metre V wind component": "v10m",
}



def request_mars_file(date):
    os.makedirs("/fast/ignored/mars_data", exist_ok=True)

    date_only = date.strftime("%Y-%m-%d")
    hour = date.strftime("%H:00:00")
    datestr = date_only.replace("-", '')+hour[:2]

    s3_bucket = 'wb-weather-archive'
    s3_key = f"mars/hres/{date.strftime('%Y/%m')}/output_{date.strftime('%Y%m%d%H')}.grib"
    output_file = f"/fast/ignored/mars_data/output_{datestr}.grib"

    try:
        S3_CLIENT.download_file(s3_bucket, s3_key, output_file)
        return
    except Exception as e:
        print(f"Not found in S3: {e}")

    template = """retrieve,
class=od,
date=%s,
expver=1,
levelist=10/20/30/50/70/100/150/200/250/300/400/500/600/700/800/850/900/925/950/1000,
levtype=pl,
param=129.128/130.128/131/132/133.128,
step=0,
stream=oper,
grid=0.25/0.25,
time=%s,
type=fc,
target="/fast/ignored/mars_data/output_%s.grib"

retrieve,
class=od,
date=%s,
expver=1,
levtype=sfc,
param=151.128/165.128/166.128/167.128,
step=0,
stream=oper,
grid=0.25/0.25,
time=%s,
type=fc,
target="/fast/ignored/mars_data/output_%s.grib"

"""
    req = template % (date_only, hour, datestr, date_only, hour, datestr)

    # only analysis is available for hours 6 and 18
    if date.hour % 12 != 0:
        req = req.replace("stream=oper", "stream=scda")

    with open("/fast/ignored/mars_data/req_%s" % datestr, "w") as f:
        f.write(req)

    # make ctrl+c happy
    time.sleep(1)
    os.system("mars /fast/ignored/mars_data/req_%s" % datestr)

    # upload to S3
    S3_CLIENT.upload_file(output_file, s3_bucket, s3_key)


def get_files(date):
    out_path = "/fast/proc/hres_rt/f000/" + "%04d%02d%02d%02d.npz" % (date.year, date.month, date.day, date.hour)
    print(f"Downloading {date} to {out_path}", flush=True)

    fs = []

    use_mars = date < datetime(2024, 3, 10, tzinfo=timezone.utc) or '--force-mars' in sys.argv

    if use_mars:
        print("Using MARS or cached MARS")
        if date.hour != 0 and '--all-mars' not in sys.argv:
            print(f"Skipping {date} because it's not a 00z run and would need a MARS request")
            return False

        print("Trying to fetch from mars")
        # try downloading from mars archive instead
        mars_file = f"/fast/ignored/mars_data/output_{date.strftime('%Y%m%d%H')}.grib"

        if not os.path.exists(mars_file):
            print(f"{mars_file} not found; requesting")
            request_mars_file(date)

        if not os.path.exists(mars_file):
            print(f"{mars_file} STILL not found")
            return False

        fs.append(mars_file)
    else:
        print("Using /huge or s3")
        s = "%02d%02d%02d0" % (date.month, date.day, date.hour)
        fn = "S" + s + "0" + s + "11"  # S03121200031212011
        for prefix in ["A3", "A4"]:
            full_fn = prefix + fn
            of = "/tmp/" + full_fn
            fs.append(of)
            if not os.path.exists(of):
                of_tmp = of + ".tmp"
                print(f"Downloading {full_fn} ", end='', flush=True)
                if date < datetime(2024, 9, 4, tzinfo=timezone.utc):
                    print(f"from /huge", flush=True)
                    if os.path.exists("/huge/proc"):
                        # if on prem and /huge exists
                        print("/huge is mounted, no need to download")
                        path = f"/huge/proc/compressed/{full_fn}"
                        os.system(f"cp {path} {of_tmp}")
                    else:
                        # on a realtime machine, we fetch it from /huge
                        username = "deploy"
                        password = "dcCTfKXEN7cPXC2EHTLD"
                        url = f"https://a.windbornesystems.com/dlnwp/ecmwf_compressed/{full_fn}"
                        print(f"getting from {url}")
                        response = requests.get(url, auth=HTTPBasicAuth(username, password))
                        if response.status_code == 200:
                            with open(of_tmp, 'wb') as file:
                                file.write(response.content)
                            print(f"saved to {of_tmp}")
                        else:
                            print(f"failed to download file: {response.status_code}")
                            exit()
                else:
                    print(f"from s3 wb-ecmwf-delivery", flush=True)
                    S3_CLIENT.download_file('wb-ecmwf-delivery', f'{full_fn}', of_tmp)
                os.rename(of_tmp, of)
            
    by_variable = {v: [] for v in varlist}

    for f in fs:
        grbs = pygrib.open(f)
        for grb in grbs:
            if grb.P1 != 0: continue # newer inputs added for doctorate
            if grb.name in varlist:
                # lons = grb.longitudes
                vals = grb.values.astype(np.float32)
                # lons.shape = vals.shape
                # lons = np.roll(lons, -720, axis=1)
                if not use_mars:  # MARS data is already shifted
                    vals = np.roll(vals, -720, axis=1)
                by_variable[grb.name].append((grb.level, vals))

            if EXPORTING and (grb.name in EXPORTED_SURFACE_VARS or (grb.name in EXPORTED_UPPER_LEVEL_VARS and grb.level in EXPORTED_LEVELS)):
                short_name = GRB_NAMES[grb.name]
                if grb.typeOfLevel == "isobaricInhPa":
                    short_name += f"_{int(grb.level)}"

                output_dir = f"{PARTIAL_EXPORT_DIR}/{date.strftime('%Y%m%d%H')}"
                output_path = os.path.join(output_dir, f"{short_name}.npy")
                if os.path.exists(output_path) and os.path.getsize(output_path) > 2**16:
                    continue

                vals = grb.values[:].astype(np.float32)
                if not use_mars:  # MARS data is already shifted
                    vals = np.roll(vals, -720, axis=1)

                if grb.name in ['Mean sea level pressure', 'Geopotential']:
                    vals = np.ma.filled(vals, fill_value=-999999)

                os.makedirs(output_dir, exist_ok=True)
                np.save(output_path, vals)

            # for k in grb.keys(): print(k, getattr(grb, k))
    # exit()

    for k in by_variable:
        by_variable[k] = sorted(by_variable[k], key=lambda x: x[0])
        by_var_levels = [x[0] for x in by_variable[k]]
        assert by_var_levels == levels_hres or len(by_variable[k]) == 1, f"len(by_variable[k])={len(by_variable[k])} len(levels_hres)={len(levels_hres)} levels_hres-by_variable[k]={set(levels_hres) - set(by_var_levels)}"
        by_variable[k] = [x[1] for x in by_variable[k]]

    print(f"Processing {date}", flush=True)
    arr = []
    for v, w in zip(pvarlist, pressure_vars):
        mean, std = norms[w]
        ohp = np.array(by_variable[v]).transpose(1, 2, 0)
        ohp = (ohp - mean[np.newaxis, np.newaxis, wh_lev]) / (std[np.newaxis, np.newaxis, wh_lev])
        # print(v,w,ohp.mean(), ohp.std())
        arr.append(ohp.astype(np.float16))
    arr = np.array(arr).transpose(1, 2, 0, 3)

    sfc = []
    for v, w in zip(svarlist, sfc_vars):
        mean, std = norms[w]
        ohp = np.array(by_variable[v])
        ohp = (ohp - mean[np.newaxis, np.newaxis]) / (std[np.newaxis, np.newaxis])
        # print(v, w, ohp.mean(), ohp.std())
        sfc.append(ohp.astype(np.float16)[0])
        # print(v, w, "ohp", ohp.shape)
    sfc = np.array(sfc).transpose(1, 2, 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    output_path_tmp = out_path.replace(".npz", ".tmp.npz")
    np.savez(output_path_tmp, pr=arr, sfc=sfc)
    os.rename(output_path_tmp, out_path)
    print(f"Saved {out_path}", flush=True)

    # Clean up
    if not use_mars and '--no-clean' not in sys.argv:
        for prefix in ["A3", "A4"]:
            full_fn = prefix + fn
            of = "/tmp/" + full_fn
            os.remove(of)


def has_data(date, var, level=None):
    if not EXPORTING:
        return False

    output_dir = os.path.join(PARTIAL_EXPORT_DIR, date.strftime("%Y%m%d%H"))
    name = GRB_NAMES[var]
    if level:
        name += f"_{level}"

    output_path = os.path.join(output_dir, f"{name}.npy")
    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 2**16:
        return False

    return True


def download_and_process(utc):
    print(f"Attempting to get ECMWF hres at {utc} (exporting={EXPORTING})", flush=True)

    has_exports = True
    has_output = os.path.exists(f"/fast/proc/hres_rt/f000/{utc.strftime('%Y%m%d%H')}.npz")

    if has_output and EXPORTING:
        for var in EXPORTED_SURFACE_VARS:
            if not has_data(utc, var):
                print(f"Missing {var}", flush=True)
                has_exports = False
                break

        if has_exports:
            for var in EXPORTED_UPPER_LEVEL_VARS:
                for level in EXPORTED_LEVELS:
                    if not has_data(utc, var, level):
                        print(f"Missing {var} at {level}", flush=True)
                        has_exports = False
                        break

    if has_output and (has_exports or not EXPORTING):
        print(f"Already processed", flush=True)
        return

    get_files(utc)


def main():
    run_from_argv(download_and_process, 'hres_rt')


if __name__ == '__main__':
    main()
