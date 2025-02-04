# %%
import sys
sys.path.append('/fast/wbhaoxing/deep')
import numpy as np
import json
import os
from datetime import datetime, timedelta, timezone
import pickle
import torch
import time
import math
from scipy.interpolate import RegularGridInterpolator, interpn, griddata, LinearNDInterpolator
from scipy.spatial import Delaunay
from hres.model import HresModel
torch.manual_seed(0)
np.random.seed(0)
from tqdm import tqdm
import requests
import xarray as xr
import pygrib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#from metpy.plots import USCOUNTIES
from data import HresDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import einops

sys.path.append('/fast/wbhaoxing/windborne')
from meteo.tools.process_dataset import download_s3_file

DEFAULT_MODEL = 'WeatherMesh'
DEFAULT_MODEL_VARIANT = 'MA1l'
BASE_VIZ_FOLDER = "/viz"
BASE_OUTPUT_FOLDER = "/fast/realtime/outputs"

metadata = None
np.set_printoptions(precision=4, suppress=True)

# %%
def load_model(checkpoint_path: str, model: HresModel):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to('cuda')
    model.eval()

pointy_ckpt_path = '/fast/ignored/runs_hres/run_May5-multivar_20240510-144336/model_keep_step133500_loss0.144.pt'
old = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True, n_out=7).half().to('cuda')
load_model(pointy_ckpt_path, old)

new_path = '/huge/deep/runs_hres/run_Sep24-oldnewslop_20241016-165042/model_step679500_loss0.206.pt' # Oct 31
new = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True, n_out=8, grid="_small").half().to('cuda')
load_model(new_path, new)

cooler_path = '/huge/deep/runs_hres/run_Oct28-coolerpacific_20241028-114757/model_step75000_loss0.227.pt'
cooler_path = '/huge/deep/runs_hres/run_Oct28-coolerpacific_20241028-114757/model_step112500_loss0.227.pt'
cooler_path = '/huge/deep/runs_hres/run_Oct28-coolerpacific_20241028-114757/model_step146000_loss0.229.pt'
cooler = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True, n_out=8, grid="_small").half().to('cuda')
load_model(cooler_path, cooler)

retro_path = '/huge/deep/runs_hres/run_Oct31-retro_20241031-122924/model_step165000_loss0.170.pt' # Nov 4
retro = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True, n_out=8, grid="_small").half().to('cuda')
load_model(retro_path, retro)

# %%
BASE = "/fast/proc/hres_consolidated/consolidated/"
STATIONS_PATH = BASE + "stations.pkl"
VAL_STATIONS_PATH = BASE + "valid_stations.pickle"
TRAIN_STATIONS_PATH = BASE + "train_stations.pickle"
MERGED_STATIONS = "/fast/ignored/merged/merged_stations.pickle"

val_stations = pickle.load(open(VAL_STATIONS_PATH, "rb"))
all_stations = pickle.load(open(STATIONS_PATH, "rb"))
all_stations_slop = pickle.load(open(MERGED_STATIONS, "rb"))

def get_lat_lon(a):
    if a[1] is None: return a[0][1]
    if a[0] is None: return a[1][1]
    r = lambda x: len(str(round(x, 4)))
    dec0 = r(a[0][1][0]) + r(a[0][1][1])
    dec1 = r(a[1][1][0]) + r(a[1][1][1])
    if dec0 > dec1:
        return a[0][1]
    else:
        return a[1][1]

val_lat_lons = [get_lat_lon(all_stations[x]) for x in val_stations]
# %%
batch_size = None
validation = True
d = HresDataset(batch_size=batch_size, validation=validation)
print(f"len(d) {len(d)}")
sampler = WeightedRandomSampler(np.array(d.weights)**(1/2.), num_samples=len(d.weights), replacement=True)
loader = DataLoader(d, batch_size=1, num_workers=2, sampler=sampler)
is_slop = np.array(d.is_slop)
all_stations_slop_coords = d.stations_latlon_ext

# %%
def interpget(src, toy, hr, a=300, b=400):
    avg = (np.load("/fast/consts/"+'/%s/%d_%d.npy' % (src, toy, hr), mmap_mode='r') - a) / b
    return avg

def make_sample(
    model: HresModel,
    era5: dict[str, torch.Tensor], 
    sample_stations: torch.Tensor,
    date: torch.Tensor,
    ret_center=False
):
    """
    era5: dict[str, torch.Tensor] has keys 'pr' and 'sfc'
        era5['pr']: torch.Tensor, shape (1, 720, 1440, 5, 28)
        era5['sfc']: torch.Tensor, shape (1, 720, 1440, 4)
    sample_stations: torch.Tensor, dtype=int, shape (1, batch)
    date: torch.Tensor, shape (1,) - the timestamp
    """
    assert ret_center == False, "only support ret_center=False for now"
    date = date[0]
    date = datetime(1970,1,1)+timedelta(seconds=int(date))
    if type(era5) == dict: assert era5["sfc"].shape[0] == 1

    interps, idxs = pickle.load(open("/fast/ignored/hres/interps%s.pickle"%model.grid, "rb"))
    statics = pickle.load(open("/fast/ignored/hres/statics%s.pickle"%model.grid, "rb"))
    nnew = interps.shape[0]

    interps2, idxs2 = pickle.load(open("/fast/ignored/hres/interps%s_old.pickle"%model.grid, "rb"))
    nold = interps2.shape[0]
    statics2 = pickle.load(open("/fast/ignored/hres/statics%s_old.pickle"%model.grid, "rb"))

    interps = np.concatenate([interps, interps2], axis=0)
    idxs = np.concatenate([idxs, idxs2], axis=0)
    statics = {x: np.concatenate([statics[x], statics2[x]], axis=0) for x in statics}
    interps = torch.tensor(interps)
    idxs = torch.tensor(idxs)
    statics = {x: torch.tensor(y) for x, y in statics.items()}

    interp = interps[sample_stations]
    idx = idxs[sample_stations]

    assert isinstance(era5, dict)
    pr = era5["pr"]
    pr_sample = pr[0, idx[..., 0], idx[..., 1], :, :]
    pr_sample = torch.sum(pr_sample * interp[:,:,:,:,None,None], axis=3)[0]

    soy = date.replace(month=1, day=1)
    toy = int((date - soy).total_seconds()/86400)
    if toy % 3 != 0:
        toy -= toy % 3
    rad = interpget("neoradiation_1", toy, date.hour)
    ang = interpget("solarangle_1", toy, date.hour, a=0, b=180/np.pi)
    rad = torch.from_numpy(rad)
    ang = torch.from_numpy(ang)
    sa = torch.sin(ang)
    ca = torch.cos(ang)
    arrs = [rad, sa, ca]

    extra = []
    for a in arrs:
        exa = a[idx[..., 0], idx[..., 1]]
        exa = torch.sum(exa * interp, axis=3)[0]
        del a
        extra.append(exa[:,:,None])

    sfc = era5["sfc"]
    sfc_sample = sfc[0, idx[..., 0], idx[..., 1], :]
    sfc_sample = torch.sum(sfc_sample * interp[:,:,:,:,None], axis=3)[0]
    sera5 = statics["era5"][sample_stations][0]

    sfc_sample = torch.cat([sfc_sample, sera5] + extra, axis=-1)

    center_pr = pr_sample[:,0]
    pr_sample = pr_sample[:, 1:]

    center_sfc = sfc_sample[:,0]

    sfc_sample = sfc_sample[:, 1:]
    sq = int(np.sqrt(sfc_sample.shape[1]))
    sfc_sample = sfc_sample.permute(0, 2, 1).view(-1, sfc_sample.shape[2], sq, sq)


    static_keys = ["mn30", "mn75"]
    modis_keys = ["modis_"+x for x in static_keys]
    static = {x: statics[x][sample_stations][0] for x in static_keys + model.do_modis*modis_keys}
    center = {x: static[x][:,0,0] for x in static_keys}
    if model.do_modis:
        for x in modis_keys:
            center[x] = torch.nn.functional.one_hot(static[x][:,0].long(), 17)
    for x in static_keys:
        sq = int(np.sqrt(static[x].shape[1]-1))
        static[x] = static[x][:,1:].view(-1, sq, sq, 3)
        if x.startswith("mn"):
            pass
        if model.do_modis:
            modis = static["modis_"+x][:, 1:].view(-1, sq, sq)#, 17)
            modis = torch.nn.functional.one_hot(modis.long(), 17)
            static[x] = torch.cat((static[x], modis), dim=3)
        static[x] = static[x].permute(0, 3, 1, 2)
    inp = {}

    sq = int(np.sqrt(pr_sample.shape[1]))
    pr_sample = pr_sample.view(-1, sq, sq, 5, pr_sample.shape[-1]).permute(0, 3, 4, 1, 2)
    inp["pr"] = pr_sample
    #print("pr shape", inp["pr"].shape, inp["sfc"].shape)
    inp["sfc"] = sfc_sample
    for k in static_keys:
        inp[k] = static[k]
    inp["center"] = torch.cat([center[x][:, None] if len(center[x].shape)==1 else center[x] for x in static_keys + model.do_modis * modis_keys], dim=1)
    inp["center"] = torch.cat([inp["center"], center_sfc], dim=1)
    inp = {x: y.half() for x, y in inp.items()}
    return inp


def compute_error(
    out: torch.Tensor, 
    data: torch.Tensor, 
    weights: torch.Tensor,
    dataset: HresDataset,
    is_normalized=True,
    signed=False,
    return_bias=False,
    n_out=8,
):
    """is_normalized=True if out is normalized (aka what the model outputs)
       data is always normalized
       signed=False outputs RMSE, signed=True outputs signed error"""
    nanrms = lambda x: torch.sqrt(torch.nansum(weights[:, None] * x**2, axis=0)/torch.sum(weights[:, None] * (~torch.isnan(x)), axis=0)).cpu().numpy()
    nanbias = lambda deltas: (torch.nansum(weights[:, None] * deltas, axis=0)/torch.sum(weights[:, None] * (~torch.isnan(deltas)), axis=0)).cpu().numpy()
    if is_normalized:
        if n_out == 8:
            delta = (out - data).float()
            stds = torch.tensor([np.sqrt(x[1][0]) for x in dataset.normalizations])
            delta_units = delta * stds[None, :].cuda()
            train = delta_units
            print(f"data nan fraction: {torch.isnan(data).sum().item() / data.numel()}")
            print(f"nan fraction: {torch.isnan(train).sum().item() / train.numel()}")
            train_rms = nanrms(train)
            train_bias = nanbias(train)
        elif n_out == 7:
            data = data[:, :-1]
            # flip u and v
            out = torch.cat([out[:, :3], out[:, 4:5], out[:, 3:4], out[:, 5:]], axis=1)
            delta = (out - data).float()
            stds = torch.tensor([np.sqrt(x[1][0]) for x in dataset.normalizations[:-1]])
            delta_units = delta * stds[None, :].cuda()
            train = delta_units
            print(f"data nan fraction: {torch.isnan(data).sum().item() / data.numel()}")
            print(f"nan fraction: {torch.isnan(train).sum().item() / train.numel()}")
            train_rms = nanrms(train)
            train_bias = nanbias(train)
        else:
            raise ValueError("n_out must be 7 or 8, what is this model??")
    else:
        norms = torch.from_numpy(np.array(dataset.normalizations)).squeeze().cuda()
        data = data * norms[:,1].sqrt() + norms[:,0]
        out[:,6] *= 0.000621371 # visibility is in miles in METAR and meters in HRRR
        train = out.cuda() - data
        print(f"data nan fraction: {torch.isnan(data).sum().item() / data.numel()}")
        print(f"nan fraction: {torch.isnan(train).sum().item() / train.numel()}")
        train_rms = nanrms(train)
        train_bias = nanbias(train)

    if not signed: 
        if return_bias:
            return train_rms, train_bias
        else:
            return train_rms
    else:
        if return_bias:
            return train.cpu().numpy(), train_bias
        else:
            return train.cpu().numpy()

# %%
grib_file = "/huge/users/haoxing/pointy/hrrr.t00z.wrfsfcf00.grib2"
grbs = pygrib.open(grib_file)
hrrr_lats, hrrr_lons = grbs[1].latlons()

# %%
POINTY_VARS = ["tmpf", "dwpf", "mslp", '10u', '10v', 'skyl1', 'vsby', 'precip']
HRRR_VARS = ["2 metre temperature", "2 metre dewpoint temperature", "MSLP (MAPS System Reduction)", "10 metre U wind component", "10 metre V wind component", "Low cloud cover", "Visibility", "Total Precipitation"]
# %%
n_lats, n_lons = hrrr_lats.shape[0], hrrr_lats.shape[1]
hrrr_grid = np.zeros((n_lats, n_lons, 2))
for i in range(n_lats):
    for j in range(n_lons):
        hrrr_grid[i, j] = [hrrr_lats[i, j], hrrr_lons[i, j]]
print(f"Computing interpolation weights for the HRRR grid...")
tri = Delaunay(hrrr_grid.reshape(-1, 2))

# %%
def is_conus(lat, lon):
    return 24 <= lat <= 50 and -125 <= lon <= -66

def compute_hrrr_error(
    tri: Delaunay,
    hrrr_data: np.ndarray,
    data: torch.Tensor,
    sample_stations: torch.Tensor,
    weights: torch.Tensor,
    signed=False,
    return_bias=False,
):
    station_lat_lons = [all_stations_slop_coords[x] for x in sample_stations[0]]
    # shape = hrrr_data.shape
    # hrrr_data[:,:,:1] = np.random.normal(size=(shape[0], shape[1], 1)) * np.sqrt(236.8514) + 287.4892
    hrrr_interp = LinearNDInterpolator(tri, hrrr_data.reshape(-1, 8))
    hrrr_vals = hrrr_interp(station_lat_lons)
    hrrr_errs = compute_error(torch.tensor(hrrr_vals), data, weights, d, is_normalized=False, signed=signed, return_bias=return_bias)
    return hrrr_errs

# %% compare old, new, and HRRR
hrrr_bucket = "noaa-hrrr-bdp-pds"
hrrr_data_path = "/huge/users/haoxing/hrrr"
# rmses = {"new": [], "old": [], "cooler": [], "hrrr": [], "retro": []}
# biases = {"new": [], "old": [], "cooler": [], "hrrr": [], "retro": []}
rmses = {"new": [], "cooler": [], "hrrr": [], "retro": []}
biases = {"new": [], "cooler": [], "hrrr": [], "retro": []}
sizes, weightses = [], []
for sample in tqdm(loader):
    era5, data, sample_stations, weights, date = sample
    station_lat_lons = [all_stations_slop_coords[x] for x in sample_stations[0]]
    is_conuses = [is_conus(lat, lon) for lat, lon in station_lat_lons]
    non_slops = ~is_slop[sample_stations[0]] & is_conuses
    non_slop_stations = sample_stations[0][non_slops].unsqueeze(0)
    data = data[0].to('cuda')
    non_slop_data = data[non_slops]
    weights = weights[0].float().to('cuda')
    non_slop_weights = weights[non_slops]
    print(f"mini batch size: {len(sample_stations[0])}, non slop size: {len(non_slop_stations[0])}")
    # hrrr
    datestr = datetime.utcfromtimestamp(date[0].item()).strftime("%Y%m%d")
    cycle = datetime.utcfromtimestamp(date[0].item()).strftime("%H")
    if os.path.exists(f"/huge/users/haoxing/hrrr/errors/{datestr}_{cycle}.npy"):
        print("HRRR error already computed, loading...")
        hrrr_rmse, hrrr_bias = np.load(f"/huge/users/haoxing/hrrr/errors/{datestr}_{cycle}.npy")
    else:
        hrrr_key = f"hrrr.{datestr}/conus/hrrr.t{cycle}z.wrfsfcf00.grib2"
        local_path = f"{hrrr_data_path}/{datestr}.t{cycle}z.wrfsfcf00.grib2"
        download_s3_file(hrrr_bucket, hrrr_key, local_path)
        hrrr_data = []
        for hrrr_var in HRRR_VARS:
            hrrr_data.append(pygrib.open(local_path).select(name=hrrr_var)[0].values)
        hrrr_data = np.stack(hrrr_data, axis=-1)
        print("Computing HRRR error...")
        hrrr_rmse, hrrr_bias = compute_hrrr_error(tri, hrrr_data, data, sample_stations, weights, signed=False, return_bias=True)
        np.save(f"/huge/users/haoxing/hrrr/errors/{datestr}_{cycle}.npy", np.array([hrrr_rmse, hrrr_bias]))
    rmses["hrrr"].append(hrrr_rmse)
    biases["hrrr"].append(hrrr_bias)
    print(f"HRRR rmse: {hrrr_rmse}")
    print(f"HRRR bias: {hrrr_bias}")

    # interpolated era5
    # era5_interp = RegularGridInterpolator((era5_lats, era5_lons), era5, )

    # old and new
    for model in [new, cooler, retro]:
        inp = make_sample(model, era5, non_slop_stations, date)
        for key in inp:
            inp[key] = inp[key].to('cuda')
        with torch.no_grad():
            out = model(inp)
            rmse, bias = compute_error(out, non_slop_data, non_slop_weights, d, n_out=model.n_out, signed=False, return_bias=True)
            model_name = "old" if model == old else "new" if model == new else "cooler" if model == cooler else "retro"
            print(f"{model_name} rmse: {rmse}")
            print(f"{model_name} bias: {bias}")
            rmses[model_name.lower()].append(rmse)
            biases[model_name.lower()].append(bias)
    for key in rmses:
        rmses_temp = np.concatenate(rmses[key])
    sizes.append(len(non_slop_stations[0]))
    # cumulative rmses and biases
    for key in rmses:
        cum_rmse = np.stack(rmses[key])
        cum_bias = np.stack(biases[key])
        sizes_np = np.array(sizes)
        sizes_7 = einops.repeat(sizes_np, 'b -> b n', n=7)
        sizes_8 = einops.repeat(sizes_np, 'b -> b n', n=8)
        if key == "old":
            cum_rmse_true = ((cum_rmse ** 2 * sizes_7).sum(axis=0) / sizes_np.sum()) ** 0.5
            cum_bias_true = (cum_bias * sizes_7).sum(axis=0) / sizes_np.sum()
            print(f"cumulative RMSEs for {key}: {cum_rmse_true}")
            print(f"cumulative biases for {key}: {cum_bias_true}")
        else:
            cum_rmse_true = ((cum_rmse ** 2 * sizes_8).sum(axis=0) / sizes_np.sum()) ** 0.5
            cum_bias_true = (cum_bias * sizes_8).sum(axis=0) / sizes_np.sum()
            print(f"cumulative RMSEs for {key}: {cum_rmse_true}")
            print(f"cumulative biases for {key}: {cum_bias_true}")