import boto3
from datetime import datetime, timedelta, timezone
import os
import pygrib
import sys
import time

from utils import *
from .data_fetch_helpers import run_from_argv

globals().update(DEFAULT_CONFIG.__dict__)

grid = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
norms, _ = load_state_norm(grid.wh_lev, DEFAULT_CONFIG)

wh_lev = [levels_full.index(l) for l in levels_ecm1]

S3_CLIENT = boto3.client('s3', region_name='us-west-2')
prefixes = ["A1", "A2"]
pvarlist = ["Geopotential", "Temperature", "U component of wind", "V component of wind", "Specific humidity"]
svarlist = ["10 metre U wind component", "10 metre V wind component", "2 metre temperature", "Mean sea level pressure"]
varlist = pvarlist + svarlist

ENSEMBLE_COUNT = 51
OUTPUT_PATH = "/fast/proc/ens_rt/"
#OUTPUT_PATH = "/huge/users/anuj/ens_rt_from_grib/"

def request_mars_file(date):
    os.makedirs("/fast/ignored/mars_data", exist_ok=True)

    template = """retrieve,
class=od,
date=%s,
expver=1,
levelist=10/20/30/50/70/100/150/200/250/300/400/500/600/700/800/850/900/925/950/1000,
levtype=pl,
number=1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50,
param=129.128/130.128/131/132/133.128,
step=0,
stream=enfo,
grid=0.25/0.25,
time=%s,
type=pf,
target="/fast/ignored/mars_data/ens_output_%s.grib"

retrieve,
class=od,
date=%s,
expver=1,
levtype=sfc,
number=1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50,
param=151.128/165.128/166.128/167.128,
step=0,
stream=enfo,
grid=0.25/0.25,
time=%s,
type=pf,
target="/fast/ignored/mars_data/ens_output_%s.grib"

"""
    date_only = date.strftime("%Y-%m-%d")
    hour = date.strftime("%H:00:00")
    datestr = date_only.replace("-", '')+hour[:2]

    req = template % (date_only, hour, datestr, date_only, hour, datestr)

    with open("/fast/ignored/mars_data/ens_req_%s" % datestr, "w") as f:
        f.write(req)

    # make ctrl+c happy
    time.sleep(1)
    os.system("mars /fast/ignored/mars_data/ens_req_%s" % datestr)

def get_files(date):
    out_path = f"{OUTPUT_PATH}/%d/f000/" + "%04d%02d%02d%02d.npz" % (date.year, date.month, date.day, date.hour)
    print(f"Downloading {date} to {out_path}", flush=True)

    fs = []

    use_mars = date < datetime(2024, 3, 10, tzinfo=timezone.utc) or '--force-mars' in sys.argv

    if use_mars:
        # if date.hour != 0:
        #     print(f"Skipping {date} because it's not a 00z run and would need a MARS request")
        #     return False

        print("Trying to fetch from mars")
        # try downloading from mars archive instead
        mars_file = f"/fast/ignored/mars_data/ens_output_{date.strftime('%Y%m%d%H')}.grib"

        if not os.path.exists(mars_file):
            print(f"{mars_file} not found; requesting")
            request_mars_file(date)

        if not os.path.exists(mars_file):
            print(f"{mars_file} STILL not found")
            return False

        fs.append(mars_file)
    else:
        s = "%02d%02d%02d" % (date.month, date.day, date.hour)
        if date <= datetime(2024,8,5,tzinfo=timezone.utc):  # when ecmwf started giving us 6z and 12z and changed the code for ens dataa
            assert date.hour not in [6,18], "We only had 0z and 12z for ens before 20240805"
            code = 'E'
        else:
            code = 'X'
        fn = code + s +"00"+ s + "001"  # eg. A1E08050000080500001 and A1X08050600080506001
        for pref in prefixes:
            full_fn = pref + fn
            of = "/tmp/" + full_fn
            fs.append(of)
            if not os.path.exists(of):
                of_tmp = of + ".tmp"
                print(f"Downloading {full_fn} ", end='', flush=True)
                if date < datetime(2024, 9, 4, tzinfo=timezone.utc):
                    print(f"our s3 bucket doesn't have data before 2024-09-04 and we torched the GCP bucket that did. use /huge/proc/weather-archive/ for init files", flush=True)                    
                else:
                    print(f"from S3", flush=True)
                    S3_CLIENT.download_file('wb-ecmwf-delivery', f'{full_fn}', of_tmp)
                os.rename(of_tmp, of)

    byens = {k: {v: [] for v in varlist} for k in range(ENSEMBLE_COUNT)}
    for f in fs:
        grbs = pygrib.open(f)
        for grb in grbs:
            if grb.name in varlist:
                # lons = grb.longitudes
                vals = grb.values.astype(np.float32)
                # lons.shape = vals.shape
                # lons = np.roll(lons, -720, axis=1)
                vals = np.roll(vals, -720, axis=1)
                byens[grb.perturbationNumber][grb.name].append((grb.level, vals))
            # for k in grb.keys(): print(k, getattr(grb, k))
    # exit()

    for e in byens:
        for k in byens[e]:
            byens[e][k] = sorted(byens[e][k], key=lambda x: x[0])
            assert [x[0] for x in byens[e][k]] == levels_ecm1 or len(byens[e][k]) == 1
            byens[e][k] = [x[1] for x in byens[e][k]]

    for e in byens:
        print(f"Processing ensemble {e} ({e + 1}/{ENSEMBLE_COUNT})", flush=True)
        arr = []
        for v, w in zip(pvarlist, pressure_vars):
            mean, std = norms[w]
            ohp = np.array(byens[e][v]).transpose(1, 2, 0)
            ohp = (ohp - mean[np.newaxis, np.newaxis, wh_lev]) / (std[np.newaxis, np.newaxis, wh_lev])
            # print(v,w,ohp.mean(), ohp.std())
            arr.append(ohp.astype(np.float16))
        arr = np.array(arr).transpose(1, 2, 0, 3)

        sfc = []
        for v, w in zip(svarlist, sfc_vars):
            mean, std = norms[w]
            ohp = np.array(byens[e][v])
            ohp = (ohp - mean[np.newaxis, np.newaxis]) / (std[np.newaxis, np.newaxis])
            # print(v, w, ohp.mean(), ohp.std())
            sfc.append(ohp.astype(np.float16)[0])
            # print(v, w, "ohp", ohp.shape)
        sfc = np.array(sfc).transpose(1, 2, 0)
        oo = out_path % e
        os.makedirs(os.path.dirname(oo), exist_ok=True)

        op = oo.replace(".npz", ".tmp.npz")
        np.savez(op, pr=arr, sfc=sfc)
        os.rename(op, oo)
        # print(k, [x[0] for x in byens[0][k]])

    # Clean up
    for pref in prefixes:
        full_fn = pref + fn
        of = "/tmp/" + full_fn
        os.remove(of)


def download_and_process(utc):
    print(f"Attempting to get ECMWF ens at {utc}", flush=True)

    already_processed_count = 0
    for ens in range(ENSEMBLE_COUNT):
        if os.path.exists(f"{OUTPUT_PATH}/{ens}/f000/{utc.strftime('%Y%m%d%H')}.npz"):
            already_processed_count += 1
            continue

    if already_processed_count == ENSEMBLE_COUNT:
        print(f"All {ENSEMBLE_COUNT} ensemble members already processed", flush=True)
        clean_old()
        return

    if already_processed_count > 0:
        print(f"{already_processed_count} ensembles already processed", flush=True)

    get_files(utc)


def clean_old():
    now = datetime.now(timezone.utc)
    for ens in range(ENSEMBLE_COUNT):
        for f in os.listdir(f"{OUTPUT_PATH}/{ens}/f000"):
            if f.endswith(".npz"):
                path = f"{OUTPUT_PATH}/{ens}/f000/{f}"
                cycle_time = datetime.strptime(f[:-4], "%Y%m%d%H").replace(tzinfo=timezone.utc)

                if (now - cycle_time) > timedelta(days=3):
                    os.remove(path)
                    print(f"Removed {path} ({(now - cycle_time).days} days old)")


def main():
    run_from_argv(download_and_process, 'ens_rt', clean_fn=clean_old)


if __name__ == '__main__':
    main()
