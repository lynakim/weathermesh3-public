# %%
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import pickle
import pygrib
import time
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RegularGridInterpolator, interp2d
import sys
sys.path.append('/fast/wbhaoxing/deep')

from utils import levels_joank, levels_medium, levels_tiny, levels_full, levels_hres, core_pressure_vars, core_sfc_vars, aggregated_sfc_vars, levels_ncarhres

pressure_vars = ["129_z", "130_t", "131_u", "132_v", "133_q"]
all_sfc_vars = [
    "034_sstk",
    "151_msl",
    "164_tcc",
    "165_10u", 
    "166_10v", 
    "167_2t",
    "168_2d",
    "246_100u",
    "247_100v",
    "121_mx2t6",
    "122_mn2t6",
    "142_lsp",
    "143_cp",
    "176_ssr",
    "201_mx2t",
    "202_mn2t",
]

ordered_sfc_vars = ["VAR_10U", "VAR_10V", "VAR_2T", "MSL"]
ordered_pr_vars = ["Z", "T", "U", "V", "Q"]

ncarhres_dir = "/huge/proc/ncarhres"

with open("/fast/consts/normalization.pickle", "rb") as f:
    normalization = pickle.load(f)

wh_lev = [levels_full.index(l) for l in levels_ncarhres]

# skipped:
# 20220414
# 20220415
# 20230518

# start = datetime(2023, 5, 19)
# end = datetime(2023, 12, 31)
#end = datetime(2024, 10, 31)

# %%

def get_ncarhres_files(date: datetime):
    dir = f"/huge/proc/ncarhres/{date.year}{date.month:02d}"
    paths = []
    for pr_var in pressure_vars:
        for fz in ["00", "06", "12", "18"]:
            uv_or_sc = "uv" if pr_var in ["131_u", "132_v"] else "sc"
            paths.append(f"{dir}/ec.oper.an.pl.128_{pr_var}.regn1280{uv_or_sc}.{date.strftime('%Y%m%d')}{fz}.nc")
        if not os.path.exists(paths[-1]):
            print(f"missing {paths[-1]}")
            paths.pop()
    for sfc_var in core_sfc_vars:
        ver = 128 if sfc_var not in ["246_100u", "247_100v",] else 228
        an_or_fc = "an" if sfc_var not in ["121_mx2t6",
                                            "122_mn2t6",
                                            "142_lsp",
                                            "143_cp",
                                            "176_ssr",
                                            "201_mx2t",
                                            "202_mn2t",
                                        ] else "fc"
        paths.append(f"{dir}/ec.oper.{an_or_fc}.sfc.{ver}_{sfc_var}.regn1280sc.{date.strftime('%Y%m%d')}.nc")
        if not os.path.exists(paths[-1]):
            print(f"missing {paths[-1]}")
            paths.pop()
    return paths

t0 = time.time()
res = 0.1
tgtlat = np.arange(90, -90.01, -res)
tgtlon = np.arange(0, 359.99, res)
Lons, Lats = np.meshgrid(tgtlon, tgtlat)
positions = np.vstack([Lats.ravel(), Lons.ravel()]).T
print(f"took {time.time()-t0:.2f}s to prepare positions")

def pad_and_interp(arr: np.ndarray, lats: list, lons: list, positions: np.ndarray) -> np.ndarray:
    assert arr.shape == (2560, 5120), arr.shape
    padded = np.zeros((2562, 5121), dtype=np.float32) + np.nan
    padded[1:-1,:-1] = arr
    padded[0,:] = padded[1,:]
    padded[-1,:] = padded[-2,:]
    padded[:,-1] = padded[:,0]
    assert np.isnan(padded).sum() == 0, np.isnan(padded).sum()
    grd = RegularGridInterpolator((lats[::-1], lons), np.flipud(padded), method='linear')
    reint = grd(positions)
    reint.shape = Lons.shape
    return reint

def pad_and_interp2d(arr: np.ndarray, lats_orig: np.ndarray, lons_orig: np.ndarray, lats_tgt: np.ndarray, lons_tgt: np.ndarray) -> np.ndarray:
    assert arr.shape == (2560, 5120), arr.shape
    padded = np.zeros((2562, 5121), dtype=np.float32) + np.nan
    padded[1:-1,:-1] = arr
    padded[0,:] = padded[1,:]
    padded[-1,:] = padded[-2,:]
    padded[:,-1] = padded[:,0]
    assert np.isnan(padded).sum() == 0, np.isnan(padded).sum()
    interp = interp2d(lons_orig, lats_orig[::-1], padded, kind='linear')
    reint = interp(lons_tgt, lats_tgt)
    assert reint.shape == Lons.shape, reint.shape
    return reint

# %%
def main(start: datetime, end: datetime):
    for date in [start + timedelta(days=i) for i in range((end-start).days+1)]:
        #try:
        paths = get_ncarhres_files(date)
        sfc_vars = defaultdict(list)
        pr_vars = {init_h: defaultdict(list) for init_h in ["00", "06", "12", "18"]}
        t00 = time.time()
        for path in paths:
            print(f"processing {path}")
            if "sfc" not in path:
                # process pr var
                ds = Dataset(path)
                init_h = path.split(".")[-2][-2:]
                lats = np.array([90] + list(ds['latitude'][:]) + [-90])
                lons = np.array(list(ds['longitude']) + [360])
                vars = ds.variables.keys()
                for v in vars:
                    if v[0].isupper():
                        var = v
                        break
                print(f"var is {var}")
                t0 = time.time()
                arr = ds[var][0].filled(np.nan)
                print(f"took {time.time()-t0:.2f}s to load var")
                assert arr.shape == (25, 2560, 5120), arr.shape
                for i in range(25):
                    t0 = time.time()
                    reint = pad_and_interp2d(arr[i], lats, lons, tgtlat, tgtlon)
                    print(f"took {time.time()-t0:.2f}s to interpolate")
                    pr_vars[init_h][var].append(reint)
            elif path.split(".")[4][4:] not in core_sfc_vars:
                # skip extra sfc vars for now
                print(f"skipping bc extra var")
                continue
            else:
                ds = Dataset(path)
                lats = np.array([90] + list(ds['latitude'][:]) + [-90])
                lons = np.array(list(ds['longitude']) + [360])
                vars = ds.variables.keys()
                for v in vars:
                    if v[0].isupper():
                        var = v
                        break
                print(f"var is {var}")
                for i, init_time in enumerate(['00', '06', '12', '18']):
                    arr = ds[var][i].filled(np.nan)
                    t0 = time.time()
                    reint = pad_and_interp2d(arr, lats, lons, tgtlat, tgtlon)
                    print(f"took {time.time()-t0:.2f}s to interpolate")
                    sfc_vars[var].append(reint)
        print(f"took {time.time()-t00:.2f}s to process date {date}")

        # normalize and save processed vars
        for i, init_h in enumerate(["00", "06", "12", "18"]):
            valid_time = date + timedelta(hours=int(init_h))
            print(f"saving {valid_time}")
            sfc, pr = [], []
            for wb_var, var in zip(core_sfc_vars, ordered_sfc_vars):
                mean, std = normalization[wb_var]
                arr = sfc_vars[var][i]
                arr = (arr - mean) / np.sqrt(std)
                sfc.append(arr.astype(np.float16))
            for wb_var, var in zip(core_pressure_vars, ordered_pr_vars):
                mean, std = normalization[wb_var]
                arr = np.stack(pr_vars[init_h][var], axis=-1)
                arr = (arr - mean[wh_lev]) / np.sqrt(std[wh_lev])
                pr.append(arr.astype(np.float16)) # (1801, 3600, 25)
            sfc = np.stack(sfc, axis=-1)
            pr = np.stack(pr, axis=-2)
            assert sfc.shape == (1801, 3600, 4), sfc.shape
            assert pr.shape == (1801, 3600, 5, 25), pr.shape

            tmp_path = f"/fast/proc/hres9km/f000/{valid_time.strftime('%Y%m%d%H')}.tmp.npz"
            np.savez(tmp_path, sfc=sfc, pr=pr)
            path = f"/fast/proc/hres9km/f000/{valid_time.strftime('%Y%m%d%H')}.npz"
            os.rename(tmp_path, path)
        # except Exception as e:
        #     print(f"error processing {date}: {e}, continuing")
        #     # write date to /fast/proc/hres9km/README.txt
        #     with open("/fast/proc/hres9km/README.txt", "a") as f:
        #         f.write(f"{date.strftime('%Y%m%d')} not processed   \n")

# %%
# start = datetime(2016, 1, 1)
# end = datetime(2016, 12, 31)
# main(start, end)

# %%

if __name__ == "__main__":
    try:
        start_str = sys.argv[1]
        end_str = sys.argv[2]
    except:
        print("Usage: python3 process_ncarhres.py start_date end_date")
        print("e.g. python3 process_ncarhres.py 20160101 20161231")
        sys.exit(1)
    
    start = datetime.strptime(start_str, "%Y%m%d")
    start = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    end = datetime.strptime(end_str, "%Y%m%d")
    end = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
    main(start, end)
# %%
# for i, init_h in enumerate(["00", "06", "12", "18"]):
#     valid_time = date + timedelta(hours=int(init_h))
#     print(f"saving {valid_time}")
#     sfc, pr = [], []
#     for var in ordered_sfc_vars:
#         sfc.append(sfc_vars[var][i])
#     for var in ordered_pr_vars:
#         pr.append(np.stack(pr_vars[init_h][var], axis=-1)) # (1801, 3600, 25)
#     sfc = np.stack(sfc, axis=-1)
#     pr = np.stack(pr, axis=-2)
#     assert sfc.shape == (1801, 3600, 4), sfc.shape
#     assert pr.shape == (1801, 3600, 5, 25), pr.shape
#     tmp_path = f"/fast/proc/hres9km/f000/{valid_time.strftime('%Y%m%d%H')}.tmp.npz"
#     np.savez(tmp_path, sfc=sfc, pr=pr)
#     path = f"/fast/proc/hres9km/f000/{valid_time.strftime('%Y%m%d%H')}.npz"

# # %%
# grbs = pygrib.open('/huge/proc/hres9km/output_201607_0a.grib')
# # %%
# grb = grbs[96].values[:]
# # %%
# import matplotlib.pyplot as plt
# # %%
# plt.imshow(grb)
# # %%
# plt.imshow(pr_vars["00"]["Z"][-1])

# # %%
# plt.hist((grb - pr_vars['00']['Z'][-1]).flatten(), bins=1000)
# plt.yscale('log')
# %%
