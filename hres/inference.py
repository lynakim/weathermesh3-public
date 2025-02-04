import sys
import os
sys.path.append('/fast/wbhaoxing/deep')
import numpy as np
import json
import pickle
import torch
import math
from scipy.interpolate import RegularGridInterpolator, interpn, griddata, LinearNDInterpolator
from hres.model import HresModel
torch.manual_seed(0)
np.random.seed(0)
from tqdm import tqdm
from types import SimpleNamespace

from utils import levels_full, levels_joank, levels_medium, interp_levels

DEFAULT_MODEL = 'WeatherMesh'
DEFAULT_MODEL_VARIANT = 'MA1l' # note this should change but I'm lazy
BASE_VIZ_FOLDER = "/viz"
BASE_OUTPUT_FOLDER = "/fast/realtime/outputs"


def read_metadata():
    with open(f"{BASE_OUTPUT_FOLDER}/{DEFAULT_MODEL}/meta.{DEFAULT_MODEL_VARIANT}.json", "r") as f:
        metadata = json.load(f)
    return metadata


def load_model(checkpoint_path: str, model: HresModel):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to('cuda')
    model.eval()


def interp(points, mode=""):
    points = [(point[0],point[1]) for point in points]
    def get_points(lat, lon, res, N, with_emb=False):
        out = []
        emb = []
        for dy in np.arange(-N+0.5, N)[::-1]:
            for dx in np.arange(-N+0.5, N):
                f = 111.111
                out.append((min(90, max(-90+1e-5, lat + dy*res/f)), (360 + lon + dx*res/(f * np.cos(lat * np.pi/180))) % 360))
                if with_emb:
                    emb.append((dy*res, dx*res))
        if with_emb:
            return out, emb
        return out
    def get_interp(pts, gridres):
        out = []
        idxs = []
        for lat, lon in pts:
            tlat = int(math.floor((90-lat)/gridres))
            blat = min(tlat+1, int(round(180/gridres))-1)
            llon = int(math.floor(lon/gridres))
            rlon = (llon+1)%int(round(360/gridres))
            v1 = np.array([(llon+1)*gridres - lon, lon - llon*gridres])
            v2 = np.array([(90 - blat*gridres)-lat, lat - (90 - tlat*gridres)])
            det = 1/(-gridres * gridres)
            vals = det * np.outer(v1, v2).flatten()
            idx = [(tlat, llon), (blat, llon), (tlat, rlon), (blat, rlon)]
            out.append(vals)
            idxs.append(idx)
        return np.array(out), np.array(idxs)
    def get_nn(pts, gridres):
        idxs = []
        for lat, lon in pts:
            lat = int(round((90-lat)/gridres))
            lat = max(0, min(lat, int(round(180/gridres))-1))
            lon = int(round(lon/gridres))%int(round(360/gridres))
            idxs.append((lat, lon))
        return np.array(idxs)
    def ohp(x):
        return max(x, -90+1e-5)
    latlongrid = []
    for point in points:
        latlongrid.append(point)
    latlongrid = np.array([latlongrid])
    paths = [("/fast/ignored/elevation/mn75.npy", "mn75", 0.5, 16, False), ("/fast/ignored/elevation/mn30.npy", "mn30", 2, 16, False), ("/fast/ignored/modis/2020.npy", "modis_mn75", 0.5, 16, True), ("/fast/ignored/modis/2020_small2.npy", "modis_mn30", 2, 16, True)]

    if mode == "":
        paths = [("/fast/ignored/elevation/mn75.npy", "mn75", 0.5, 16, False), ("/fast/ignored/elevation/mn30.npy", "mn30", 2, 16, False)]
        paths += [("/fast/ignored/modis/2020.npy", "modis_mn75", 0.5, 16, True), ("/fast/ignored/modis/2020_small2.npy", "modis_mn30", 2, 16, True)]
    elif mode == "_small":
        paths = [("/fast/ignored/elevation/mn75.npy", "mn75", 0.5, 8, False), ("/fast/ignored/elevation/mn30.npy", "mn30", 2, 8, False)]
        paths += [("/fast/ignored/modis/2020.npy", "modis_mn75", 0.5, 8, True), ("/fast/ignored/modis/2020_small2.npy", "modis_mn30", 2, 8, True)]
    interps = []
    idxs = []
    statics = {}
    ex = []
    for f in ["topography.npy", "soil_type.npy", "land_mask.npy"]:
        arr = np.load("/fast/consts/"+f)
        arr = np.concatenate((arr, arr[:, 0][:,None]), axis=1)
        x = np.arange(90, -90.01, -0.25)
        y = np.arange(0, 360.01, 0.25)
        sint = RegularGridInterpolator((x, y), arr, method="linear" if arr.dtype.kind == 'f' else "nearest")
        ex.append(sint)
    Ohps = []
    for latlon in latlongrid.reshape(-1, 2):
        lat, lon = latlon
        if lon < 0: lon += 360
        pts, emb = get_points(lat, lon, 20, 8, with_emb=True)
        emb = np.array([(0.,0.)]+emb)
        ohps = []
        for sint in ex:
            try: oh = sint([(lat,lon)]+pts)
            except:
                print(lat, lon)
                import traceback
                traceback.print_exc()
                exit()
            ohps.append(oh)
        ohps = np.array(ohps).T
        ohps = np.concatenate((ohps, emb), axis=1)
        ohps[:, 0] /= 1000. * 9.8
        ohps[:, 1] /= 7.
        ohps[:, 3] /= 20.
        ohps[:, 4] /= 20.
        Ohps.append(ohps)
        interp, idx = get_interp([(ohp(lat), lon)]+pts, 0.25)
        interps.append(interp)
        idxs.append(idx)
    main_interps = np.array(interps).astype(np.float32)
    main_idxs = np.array(idxs).astype(np.int32)
    statics["era5"] = np.array(Ohps).astype(np.float16)
    for path, name, ores, oN, onehot in paths:
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception as e:
            print(f"Path {path} didn't load correctly", flush=True)

            raise e
        res = 180/arr.shape[0]
        static = []
        for latlon in latlongrid.reshape(-1, 2):
            lat, lon = latlon
            if lon < 0: lon += 360
            pts, emb = get_points(lat, lon, ores, oN, with_emb=True)
            pts = [(ohp(lat), lon)] + pts
            emb = np.array([(0.,0.)]+emb)
            if not onehot:
                interp, idxs = get_interp(pts, res)
                gath = arr[idxs[...,0], idxs[...,1]]
                vals = np.sum(gath * interp, axis=1)
                hh = np.hstack((vals[:,None], emb))
                cent = hh[0,0].copy()
                hh[:,0] = (hh[:, 0] - cent)*(1./100)
                hh[:, 1:] /= 20.
                hh[0, 0] = cent*(1./1000)
                static.append(hh.astype(np.float16))
            else:
                idxs = get_nn(pts, res)
                gath = arr[idxs[...,0], idxs[...,1]]
                fill = 17
                if lat < -65:
                    fill = 15
                gath[gath == -1] = fill
                gath -= 1
                assert gath.min() >= 0
                assert gath.max() < 17
                static.append(gath.astype(np.int8))
        statics[name] = np.array(static)
    return main_interps, main_idxs, statics


def interpget(src, toy, hr, a=300, b=400):
    avg = (np.load("/fast/consts/"+'/%s/%d_%d.npy' % (src, toy, hr), mmap_mode='r') - a) / b
    return avg


def build_input(model,era5,interp,idx,statics,date):
    #RADIATION
    soy = date.replace(month=1, day=1)
    toy = int((date - soy).total_seconds()/86400)
    toy -= toy % 3
    rad = interpget("neoradiation_1", toy, date.hour)
    ang = interpget("solarangle_1", toy, date.hour, a=0, b=180/np.pi)
    idx0, idx1 = idx[..., 0], idx[..., 1]
    rad = rad[idx0, idx1]
    ang = ang[idx0, idx1]
    rad = np.sum(rad * interp, axis=3)[0]
    ang = np.sum(ang * interp, axis=3)[0]
    rad = rad[:,:,None]
    ang = ang[:,:,None]
    sa = np.sin(ang)
    ca = np.cos(ang)
    extra = [rad, sa, ca]
    #END RADIATION  

    #PRESSURE
    pr_sample = era5["pr"][0]
    pr_sample = pr_sample[:, 1:]
    sq = int(np.sqrt(pr_sample.shape[1]))
    pr_sample = pr_sample.reshape(-1, sq, sq, 5, 28).transpose(0, 3, 4, 1, 2)
    #END PRESSURE
    #SURFACE
    sfc_sample = era5["sfc"][0]
    sera5 = statics["era5"]
    sfc_sample = np.concatenate([sfc_sample, sera5] + extra, axis=-1)
    center_sfc = sfc_sample[:,0]
    sfc_sample = sfc_sample[:, 1:]
    sq = int(np.sqrt(sfc_sample.shape[1]))
    sfc_sample = sfc_sample.transpose(0, 2, 1).reshape(-1, sfc_sample.shape[2], sq, sq)
    #END SURFACE

    #STATIC
    static_keys = ["mn30", "mn75"]
    modis_keys = ["modis_"+x for x in static_keys]
    static = {x: statics[x] for x in static_keys + model.do_modis*modis_keys}
    center = {x: static[x][:,0,0] for x in static_keys}
    if model.do_modis:
        for x in modis_keys:
            center[x] = np.eye(17)[static[x][:,0].astype(int)]
    for x in static_keys:
        sq = int(np.sqrt(static[x].shape[1]-1))
        static[x] = static[x][:,1:].reshape(-1, sq, sq, 3)
        if x.startswith("mn"):
            pass
        if model.do_modis:
            modis = static["modis_"+x][:, 1:].reshape(-1, sq, sq)
            modis = np.eye(17)[modis.astype(int)]
            static[x] = np.concatenate((static[x], modis), axis=3)
        static[x] = static[x].transpose(0, 3, 1, 2)
    #END STATIC

    inp = {}
    inp["pr"] = pr_sample
    inp["sfc"] = sfc_sample
    for k in static_keys:
        inp[k] = static[k]
    inp["center"] = np.concatenate([center[x][:, None] if len(center[x].shape)==1 else center[x] for x in static_keys + model.do_modis * modis_keys], axis=1)
    inp["center"] = np.concatenate([inp["center"], center_sfc], axis=1)
    inp = {x: y.astype(np.float16) for x, y in inp.items()}
    return inp

def load_statics(grid: str = "") -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    interps, idxs = pickle.load(open("/fast/ignored/hres/interps%s.pickle"%grid, "rb"))
    statics = pickle.load(open("/fast/ignored/hres/statics%s.pickle"%grid, "rb"))

    return interps, idxs, statics

def get_point_data_nb(
    model: HresModel, 
    points: list[list[float]], 
    era5: np.ndarray, 
    valid_time,
):
    print("Interpolating...")
    interps, idxs, statics = interp(points, mode=model.grid)

    interps = np.expand_dims(interps, axis=0)
    idxs = np.expand_dims(idxs, axis=0)
    meta = read_metadata()
    levels = meta["levels"]
    n_levels = len(levels)
    inp = {
        "center": [],
        "pr": [],
        "sfc": [],
        "mn30": [],
        "mn75": [],
    }

    valid = valid_time.isoformat()

    # #NORMALIZATION
    with open("/fast/consts/normalization.pickle", "rb") as f:
        normalization = pickle.load(f)
    wanted_indexes = [levels_full.index(x) for x in levels]
    pressure_vars = meta["pressure_vars"]
    sfc_vars = meta["sfc_vars"][:4]

    assert len(pressure_vars) == 5, len(pressure_vars)
    assert len(sfc_vars) == 4, len(sfc_vars)
    assert len(era5.shape) == 3, era5.shape

    era5 = era5[idxs[..., 0], idxs[..., 1], :]
    era5 = np.sum(era5 * interps[:,:,:,:,None], axis=3)[0]

    sfc_means, sfc_stds = [], []
    for var in sfc_vars:
        mean, std = normalization[var]
        sfc_means.append(mean)
        sfc_stds.append(std)
    pr_means, pr_stds = np.zeros((5, n_levels)), np.zeros((5, n_levels))
    for i, var in enumerate(pressure_vars):
        mean, std = normalization[var]
        pr_means[i] = mean[wanted_indexes]
        pr_stds[i] = std[wanted_indexes]
    all_means = pr_means.flatten().tolist() + sfc_means
    all_stds = pr_stds.flatten().tolist() + sfc_stds
    # print(f"all means: {all_means}, all stds: {all_stds}")
    # print(f"len(all_means): {len(all_means)}, len(all_stds): {len(all_stds)}")

    n_core_vars = len(pressure_vars) * n_levels + len(sfc_vars)
    assert n_core_vars in [144, 129], "expecting 144 or 129 core variables for 28 or 25 levels"
    for i in range(n_core_vars):
        era5[:,:,i] = (era5[:,:,i] - all_means[i]) / np.sqrt(all_stds[i])

    # interp levels if not 28 levels
    if n_levels != 28:
        assert n_levels == 25, "assume either 28 or 25 levels for now"
        # eh no real mesh but this should work
        mesh = SimpleNamespace(n_pr_vars=5)
        era5 = interp_levels(torch.from_numpy(era5), mesh, levels_joank, levels_medium).numpy()
        assert era5.shape[-1] == 147, era5.shape

    sfc_era5 = era5[:,:,-7:-3] # (n_points, 257, 4)
    pr_era5 = era5[:,:,:-7] # (n_points, 257, 140)

    era5 = {"sfc": sfc_era5, "pr": pr_era5.reshape(pr_era5.shape[0], pr_era5.shape[1], 5, 28)}
    era5 = {x: np.expand_dims(y, axis=0) for x, y in era5.items()}  # keys: pr, sfc

    print("Building input...")
    inp = build_input(model, era5, interps, idxs, statics, valid_time)

    full = inp["pr"].shape[0]
    batch_size = 500
    outs = []
    print("Running model...")
    with torch.no_grad():
        for b in tqdm(range(0, full, batch_size)):
            end = min(full, b + batch_size)
            inpx = {x: torch.tensor(y[b:end]).half().to('cuda') for x, y in inp.items()}
            #print(f"inp['pr'].shape: {inp['pr'].shape}")
            out = model(inpx).cpu().numpy()
            outs.append(out)
    out = np.concatenate(outs, axis=0)

    def n(arr, norm):
        arr = arr.astype(np.float32)
        return arr * np.sqrt(normalization[norm][1][0]) + normalization[norm][0][0]
    
    num_frames = 1
    num_points = len(points)
    tmp = (n(out[:,0], '167_2t')-273.15).reshape(num_frames, num_points, -1)
    dpt = (n(out[:,1], '168_2d')-273.15).reshape(num_frames, num_points, -1)
    mslp = (n(out[:, 2], '151_msl')/100).reshape(num_frames, num_points, -1)
    ucomp = (n(out[:,3], '165_10u') * 1.94).reshape(num_frames, num_points, -1)
    vcomp = (n(out[:,4], '166_10v') * 1.94).reshape(num_frames, num_points, -1)
    response = {}
    forecasts = []
    for point in points:
        point_forecasts = []
        for i in range(1):
            forecast = {}
            forecast['time'] = valid
            forecast['temperature_2m'] = round(float(tmp[i][points.index(point)]),1)
            forecast['dewpoint_2m'] = round(float(dpt[i][points.index(point)]),1)
            forecast['pressure_msl'] = round(float(mslp[i][points.index(point)]),1)
            forecast['wind_u_10m'] = round(float(ucomp[i][points.index(point)]),1)
            forecast['wind_v_10m'] = round(float(vcomp[i][points.index(point)]),1)
            point_forecasts.append(forecast)
        forecasts.append(point_forecasts)
    response['forecasts'] = forecasts
    return response