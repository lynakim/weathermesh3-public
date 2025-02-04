import numpy as np

lats = np.arange(90, -90.01, -0.25)
lons = np.arange(0,359.99,0.25)
Lons, Lats = np.meshgrid(lons, lats)

coslat = np.cos(Lats*np.pi/180)
print(coslat)

import sys
sys.path.append("..")
from utils import *
import xarray as xr

print = builtins.print

globals().update(NeoDatasetConfig(WEATHERBENCH=1).__dict__)

with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
    norm = pickle.load(f)

#print(norm["129_z"][1].shape)
#exit()

hres_dataset = xr.open_zarr('gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr')
era5_dataset = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
#full_dataset = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2")
Rms = []

def cmp(d):
    nix = to_unix(d)

    hres_data = hres_dataset.sel(time=np.datetime64(nix, 's'), level=500, prediction_timedelta=np.timedelta64(1,'D'))
    hres_gph = hres_data["geopotential"]
    #print("valid", hres_gph)

    #era5_dataset = xr.open_zarr('gs://weatherbench2/datasets/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
    era5_data = era5_dataset.sel(time=np.datetime64(nix + 86400, 's'), level=500)
    era5_gph = era5_data["geopotential"]
    #print("valid", era5_gph)
    #print("hey", hres_gph.shape, era5_gph.shape)

    #full_data = full_dataset.sel(time=np.datetime64(nix, 's'), level=500)
    #full_gph = full_data["geopotential"]

    delta = hres_gph - era5_gph
    rms = np.sqrt(np.average(np.square(delta), weights=coslat))
    #delta_full = hres_gph - full_gph
    #print("Shp", delta.shape)
    #print("rms", np.sqrt(np.mean(np.square(delta))).to_numpy())
    #print(d, "rms", rms)
    #print("rms_full", np.sqrt(np.mean(np.square(delta_full))).to_numpy())

    return rms

#cmp(datetime(2017, 6, 21, 0))
d = datetime(2022, 8, 1, 0)
while True:
    Rms.append(cmp(d))
    print(d, "last %.2f so far %.2f pm %.2f" % (Rms[-1], np.sqrt(np.mean(np.square(Rms))), np.std(Rms)/(len(Rms)**0.5)) )
    if d >= datetime(2022, 12, 30): break
    d += timedelta(hours=12)
