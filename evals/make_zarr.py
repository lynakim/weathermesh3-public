import numpy as np
import os
import json
import sys
sys.path.append("..")
from utils import *
import xarray as xr

globals().update(NeoDatasetConfig(WEATHERBENCH=1).__dict__)

base = "/fast/evaluation/TardisNeoL2FT_289M//outputs/"
#2020092012+24.Csyq.npy"

ls = os.listdir(base)
ls = sorted([x for x in ls if x.endswith(".npy") and "+24" in x])
"""
nixs = []
for d in ls:
    d = d.split("+")[0]
    d = datetime(int(d[0:4]), int(d[4:6]), int(d[6:8]), int(d[8:10]), tzinfo=timezone.utc)
    nixs.append(np.datetime64(d))
"""

meta = ls[0].split(".")[-2]
meta = base + "meta.%s.json" % meta
meta = json.load(open(meta))

lats = meta["lats"]
lats.append(-90)
lons = meta["lons"]
lons = [x+360 if x < 0 else x for x in lons]

"""
del meta["lats"]
del meta["lons"]
pprint(meta)
"""

Levels = [500, 700, 850]

def mk(fn):
    d = fn
    d = d.split("+")[0]
    d = datetime(int(d[0:4]), int(d[4:6]), int(d[6:8]), int(d[8:10]))
    d64 = np.datetime64(d)
    f = np.load(base+fn)
    dic = {}
    for pr, nam in zip(meta["pressure_vars"], cloud_pressure_vars):
        out = np.zeros((len(Levels), 721, 1440), dtype=np.float32)
        for i, lev in enumerate(Levels):
            #print("hey", len(meta["full_varlist"]), f.shape)
            x = f[:, :, meta["full_varlist"].index(pr+"_"+str(lev))]
            #print(nam, i, lev, x.mean())
            if nam == "geopotential" and 0:
                print("hey", x.mean(), i, lev, "inbig", meta["full_varlist"].index(pr+"_"+str(lev)))
            out[i] = np.concatenate((x, x[-1][np.newaxis,:]))
        dic[nam] = (["time", "prediction_timedelta", "level", "latitude", "longitude"], out[np.newaxis, np.newaxis])
    for sf, nam in zip(meta["sfc_vars"], cloud_sfc_vars):
        x = f[:, :, meta["full_varlist"].index(sf)]
        out = np.zeros((721, 1440), dtype=np.float32)
        out[:] = np.concatenate((x, x[-1][np.newaxis, :]))
        dic[nam] = (["time", "prediction_timedelta", "latitude", "longitude"], out[np.newaxis, np.newaxis])

    """
    nb = 0
    for k in dic:
        nb += dic[k].nbytes
    print(nb/1e6, "MB")
    """

    return d64, dic

#mk(ls[0]); exit()

for i, d in enumerate(ls):
    print("heyyyy", i, d)
    ohp = mk(d)

    ds = xr.Dataset(
            data_vars=ohp[1],
            coords={"longitude": lons, "latitude": lats, "level": Levels, "prediction_timedelta": [np.timedelta64(1,"D")], "time": [ohp[0]]},
        )

    enc = {}
    for v in ohp[1]:
        enc[v] = {"chunks": tuple([-1 for _ in range(len(ohp[1][v][1].shape))])}
    #encoding={"xc": {"chunks": (-1, -1)}, "yc": {"chunks": (-1, -1)}},
    enc['time'] = {
            'units': 'seconds since 1970-01-01'
        }
    if i == 0:
        ds.to_zarr("temp", encoding=enc)
    else:
        ds.to_zarr("temp", append_dim="time")#, encoding=enc)

final = xr.open_zarr("temp")
print("final")
print(final)
print("geopotential", final.geopotential.shape)

#out = [mk(x) for x in ls[:2]]
#Dic = {k: [] for k in out[0].keys()}
