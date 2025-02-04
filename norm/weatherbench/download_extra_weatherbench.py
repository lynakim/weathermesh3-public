import sys
sys.path.append('..')
import xarray as xr
from utils import *
from functools import partial
import os

grid = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
norms, _ = load_state_norm(grid.wh_lev, grid)

dataset = None

# https://weatherbench2.readthedocs.io/en/latest/data-guide.html

# dates = sorted(list(set(dates2) - set(dates)))
#print("got", len(dates)); exit()
dates = get_dates([(datetime(1976, 12, 1), datetime(1977, 6, 1), timedelta(hours=1))])
vars = ['201_mx2t', '202_mn2t']

if max(dates) > datetime(2023, 1, 1).replace(tzinfo=timezone.utc):
    print("Limiting till 2023-01-01, Weatherbench data isn't available after that, use proc_data2.py instead.")
    dates = [d for d in dates if d < datetime(2023, 1, 1).replace(tzinfo=timezone.utc)]

def fetch(date, var):
    cloudname = ncar2cloud_names[var]
    global dataset
    nix = to_unix(date)

    base = f"{PROC_PATH}/era5/extra/%s/%04d%02d/" % (var, date.year, date.month)
    os.makedirs(base, exist_ok=True)
    cp = base + "%d.npz" % nix
    if os.path.exists(cp):
        print(f"{var} already done for {date}")
        return

    t0 = time.time()
    #s = random.random()*1
    #time.sleep(s)

    if dataset is None:
        print("initializing dataset")
        #dataset = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2")
        dataset = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr")

    data = dataset.sel(time=np.datetime64(nix, 's'))
    # if not is_sfc:
    #     mean, std2 = norms[varname]
    #     ohp = data[cloudname][grid.wh_lev, :720, :].transpose('latitude', 'longitude', 'level').to_numpy()
    #     ohp = (ohp - mean[np.newaxis, np.newaxis, grid.wh_lev])/(std2[np.newaxis, np.newaxis, grid.wh_lev])
    
    mean, std2 = norms[var]
    ohp = data[cloudname][:720, :].to_numpy()
    if var in log_vars:
        ohp = np.log(np.maximum(ohp + 1e-7,0))
    ohp = (ohp - mean[np.newaxis, np.newaxis])/(std2[np.newaxis, np.newaxis])
    ohp = ohp.astype(np.float16)
    assert var in vars_with_nans or not np.isnan(ohp).any(), "NaNs in ohp"
    dt = time.time()-t0
    print(f"took {dt:.2f}s to download {var} for {date}")
    cpt = cp.replace(".npz", ".tmp.npz")
    np.savez(cpt, x=ohp)
    os.rename(cpt, cp)

pool = multiprocessing.Pool(32)
#pool.map(fetch, dates)
for var in vars:
    assert var not in core_pressure_vars + core_sfc_vars, "core vars should be downloaded with the core script"
    assert var in ncar2cloud_names, f"var {var} doesn't have a mapping to a cloud name"

for var in vars:
    print(f"Downloading {var}")
    s = list(tqdm(pool.imap_unordered(partial(fetch, var=var), dates), total=len(dates)))
    print(f"Finished downloading {var}")
