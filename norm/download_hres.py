import sys
sys.path.append("..")
import xarray as xr
from utils import *
import random

#grid = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
norms, _ = load_state_norm(list(range(len(levels_full))), DEFAULT_CONFIG)

levlev = [  50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925,
       1000]
wh_lev = [levels_full.index(l) for l in levlev]

dataset = None

dates = get_dates([(datetime(1971, 1, 1), datetime(2021, 1, 1), timedelta(hours=6))])
dates = get_dates([(datetime(1997, 1, 1), datetime(2021, 1, 1), timedelta(hours=3))])
dates2 = get_dates([(datetime(2008, 1, 1), datetime(2021, 1, 1), timedelta(hours=1))])

dates = sorted(list(set(dates2) - set(dates)))
#print("got", len(dates)); exit()
dates = get_dates([(datetime(2021, 1, 1), datetime(2023, 1, 1), timedelta(hours=3))])
dates = get_dates([(datetime(2016, 1, 1), datetime(2023, 1, 6), timedelta(hours=12))])
dates = dates[::-1]
#dates = get_dates([(datetime(2020, 1, 1), datetime(2021, 1, 1), timedelta(hours=12))])

pressure_vars = ["129_z", "130_t", "131_u", "132_v", "133_q"]
cloud_pressure_vars = ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity"]

cloud_sfc_vars = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure"]
sfc_vars = ["165_10u", "166_10v", "167_2t", "151_msl"]

def fetch(date):
    global dataset
    nix = to_unix(date)

    base = f"{PROC_PATH}/hres_debug/f000/%04d%02d/" % (date.year, date.month)
    os.makedirs(base, exist_ok=True)
    cp = base + "%d.npz" % nix
    if os.path.exists(cp):
        #print("already done", date)
        return

    t0 = time.time()
    s = random.random()*15
    time.sleep(s)

    if dataset is None:
        print("initializing dataset")
        #dataset = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2")
        dataset = xr.open_zarr('gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr')

    data = dataset.sel(time=np.datetime64(nix, 's'), prediction_timedelta=timedelta(hours=0))
    lol = []
    for v, w in zip(cloud_pressure_vars, pressure_vars):
        #print("er", v)
        mean, std2 = norms[w]
        ohp = data[v][:, 1:, :].transpose('latitude', 'longitude', 'level').to_numpy()
        ohp = (ohp - mean[np.newaxis, np.newaxis, wh_lev])/(std2[np.newaxis, np.newaxis, wh_lev])
        lol.append(ohp.astype(np.float16))
    xpr = np.array(lol).transpose(1, 2, 0, 3)[::-1,...]
    #print("hey", xpr.shape)
    #print("mean", np.mean(xpr, axis=(0,1)))
    #print("std", np.std(xpr, axis=(0,1)))
    
    lol = []
    for v, w in zip(cloud_sfc_vars, sfc_vars):
        mean, std2 = norms[w]
        ohp = data[v][:720, :].to_numpy()
        ohp = (ohp - mean[np.newaxis, np.newaxis])/(std2[np.newaxis, np.newaxis])
        lol.append(ohp.astype(np.float16)[0])
    xsfc = np.array(lol).transpose(1, 2, 0)[::-1,...]

    if np.isnan(xpr).sum() != 0 or np.isnan(xsfc).sum() != 0:
        print("uhhhhhhh there's NaNs. not saving to prevend spread of STDs")
        return
    #print("hey2", xsfc.shape)
    #print("mean", np.mean(xsfc, axis=(0,1)))
    #print("std", np.std(xsfc, axis=(0,1)))
    dt = time.time()-t0
    print("took", dt, "slept", s)
    cpt = cp.replace(".npz", ".tmp.npz")
    np.savez(cpt, pr=xpr, sfc=xsfc)
    os.rename(cpt, cp)

pool = multiprocessing.Pool(12)
pool.map(fetch, dates)

#fetch(dates[0])
