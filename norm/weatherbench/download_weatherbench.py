import sys
sys.path.append('..')
from gen1.utils import NeoDatasetConfig
import xarray as xr
from utils import *
from meshes import LatLonGrid

grid = LatLonGrid(subsamp=1, levels=levels_medium)
norms, _ = load_state_norm(grid.wh_lev, NeoDatasetConfig())

dataset = None

dates = get_dates([(datetime(1971, 1, 1), datetime(2021, 1, 1), timedelta(hours=6))])
dates = get_dates([(datetime(1997, 1, 1), datetime(2021, 1, 1), timedelta(hours=3))])
dates2 = get_dates([(datetime(2008, 1, 1), datetime(2021, 1, 1), timedelta(hours=1))])

dates = sorted(list(set(dates2) - set(dates)))
#print("got", len(dates)); exit()
dates = get_dates([(datetime(1988, 1, 4), datetime(1988, 1, 5), timedelta(hours=1))])
save_to_huge = True
huge_redownload = True
#assert not save_to_huge or time.time() < 1735956226

pressure_vars = ["129_z", "130_t", "131_u", "132_v", "133_q"]
cloud_pressure_vars = ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity"]

cloud_sfc_vars = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure"]
sfc_vars = ["165_10u", "166_10v", "167_2t", "151_msl"]

def fetch(date):
    global dataset
    nix = to_unix(date)

    base = f"{PROC_PATH}/neo_%d_%d/%04d%02d/" % (grid.subsamp, grid.n_levels, date.year, date.month)
    os.makedirs(base, exist_ok=True)
    cp = base + "%d.npz" % nix
    if os.path.exists(cp):
        print("already done", date)
        return
    if save_to_huge:
        cp = f'/huge/proc/era5/f000/%04d%02d/%d.npz' % (date.year, date.month, nix)
        if not huge_redownload and os.path.exists(cp):
            print('already saved to huge', date)
            return
        os.makedirs(f'/huge/proc/era5/f000/%04d%02d/' % (date.year, date.month), exist_ok=True)

    t0 = time.time()
    s =  0 #random.random()*15
    #time.sleep(s)

    if dataset is None:
        print("initializing dataset")
        #dataset = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2")
        dataset = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr")

    data = dataset.sel(time=np.datetime64(nix, 's'))
    lol = []
    for v, w in zip(cloud_pressure_vars, pressure_vars):
        #print("er", v)
        mean, std2 = norms[w]
        ohp = data[v][grid.wh_lev, :720, :].transpose('latitude', 'longitude', 'level').to_numpy()
        ohp = (ohp - mean[np.newaxis, np.newaxis, grid.wh_lev])/(std2[np.newaxis, np.newaxis, grid.wh_lev])
        lol.append(ohp.astype(np.float16))
        assert not np.isnan(ohp).any(), "NaNs in ohp"
    xpr = np.array(lol).transpose(1, 2, 0, 3)
    
    lol = []
    for v, w in zip(cloud_sfc_vars, sfc_vars):
        mean, std2 = norms[w]
        ohp = data[v][:720, :].to_numpy()
        ohp = (ohp - mean[np.newaxis, np.newaxis])/(std2[np.newaxis, np.newaxis])
        lol.append(ohp.astype(np.float16)[0])
        assert not np.isnan(ohp).any(), "NaNs in ohp"
    xsfc = np.array(lol).transpose(1, 2, 0)
    dt = time.time()-t0
    print("took", dt, "slept", s)
    cpt = cp.replace(".npz", ".tmp.npz")
    np.savez(cpt, pr=xpr, sfc=xsfc)
    os.rename(cpt, cp)

# check both fast and huge to confirm all your processing ahs happened
def check_missing_dates():
    dates = get_dates([(datetime(1970, 1, 1), datetime(2024, 5, 1), timedelta(hours=1))])
    dates = dates[::-1]
    # check if all dates are in cp = f'/fast/proc/era5/f000/%04d%02d/%d.npz' % (date.year, date.month, nix) or else cp = f'/huge/proc/era5/f000/%04d%02d/%d.npz' % (date.year, date.month, nix)
    dates_not_found = []
    year = None
    for date in dates:
        nix = to_unix(date)
        cp = f'/fast/proc/era5/f000/%04d%02d/%d.npz' % (date.year, date.month, nix)
        if not os.path.exists(cp):
            cp = f'/huge/proc/era5/f000/%04d%02d/%d.npz' % (date.year, date.month, nix)
            if not os.path.exists(cp):
                dates_not_found.append(date)
        if year is None:
            year = date.year
        elif year != date.year:
            # write to logfile
            with open('dates_not_found.txt', 'a') as f:
                f.write(f"by {year}, {len(dates_not_found)} dates not found\n")
                for date in dates_not_found:
                    f.write(f"{date}\n")
            year = date.year
            dates_not_found = []
    with open('dates_not_found.txt', 'a') as f:
        f.write(f"by {year}, {len(dates_not_found)} dates not found\n")
        for date in dates_not_found:
            f.write(f"{date}\n")
    print(f"not found {len(dates_not_found)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            # Parse date from command line argument in format YYYYMMDD
            date_str = sys.argv[1]
            target_date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print("Error: Date must be in format YYYYMMDD (e.g., 20240315)")
            exit()
        # Get all hours for that day
        dates = get_dates([(target_date, target_date + timedelta(days=1), timedelta(hours=1))])
        
    pool = multiprocessing.Pool(16)
    pool.map(fetch, dates)
    print(f"done {len(dates)} dates from {dates[0]} to {dates[-1]}")

    # check_missing_dates()
