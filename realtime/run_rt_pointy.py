import argparse
import boto3
from datetime import datetime, timedelta, timezone
import gc
import glob
import json
import math
import numpy as np
import os
import pickle
import queue
from scipy.interpolate import RegularGridInterpolator
import sys
import threading
import time
import torch
from types import SimpleNamespace
from typing import Optional
sys.path.append('../dlnwp_rt')
sys.path.append('/`fast/wbhaoxing')

from hres.model import HresModel
from hres.inference import load_model
from utils import levels_full, levels_medium, levels_joank, interp_levels
from realtime.consts import DEFAULT_MODEL_HASH

#BASE_OUTPUT_FOLDER = "/huge/deep/realtime/outputs"
BASE_OUTPUT_FOLDER = "/fast/realtime/outputs"
S3_UPLOAD = True

TARGET_FORECAST_HOUR = 21*24 + 6
FORECAST_HOUR_STEP = 6
S3_CLIENT = boto3.client('s3')
BUCKET = 'wb-dlnwp'

metadata = None
model = None
upload_queue = queue.Queue()
upload_thread = None

def load_pointy():
    global model
    if model is not None:
        return
    
    # "old" pointy: in operation until 2024-11-08
    # model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True, n_out=7).half()
    # save_path = '/fast/ignored/runs_hres/run_May5-multivar_20240510-144336/model_keep_step133500_loss0.144.pt'

    save_path = '/huge/deep/runs_hres/run_Oct28-coolerpacific_20241028-114757/model_step146000_loss0.229.pt'
    model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True, n_out=8, grid="_small").half().to('cuda')
    load_model(save_path, model)

def read_metadata():
    global metadata
    if metadata is None:
        old_path = f"{BASE_OUTPUT_FOLDER}/{WM_MODEL}/meta.{MODEL_HASH}.json"
        if os.path.exists(old_path):
            with open(f"{BASE_OUTPUT_FOLDER}/{WM_MODEL}/meta.{MODEL_HASH}.json", "r") as f:
                metadata = json.load(f)
        else:
            # new path is BASE_OUTPUT_FOLER/WM_MODEL/DATESTR/det/meta.json
            some_date = os.listdir(f"{BASE_OUTPUT_FOLDER}/{WM_MODEL}")[0]
            with open(f"{BASE_OUTPUT_FOLDER}/{WM_MODEL}/{some_date}/det/meta.{MODEL_HASH}.json", "r") as f:
                metadata = json.load(f)
    return metadata

def interpget_cached(neocache, src, toy, hr, a=300, b=400):
        if src not in neocache:
            neocache[src] = {}

        def load(xx, hr):
            if (xx,hr) in neocache[src]:
                return neocache[src][(xx, hr)]
            f = torch.FloatTensor
            ohp = f(((np.load("/fast/consts/"+'/%s/%d_%d.npy' % (src, xx, hr)) - a) / b))
            neocache[src][(xx,hr)] = ohp
            return ohp
        avg = load(toy, hr)
        return neocache, avg

def build_input(model,era5,interp,idx,statics,date,primary_compute='cuda'):
        assert era5["sfc"].shape[0] == 1
        if model.do_pressure:
            pr = era5["pr"].to(primary_compute)
            pr_sample = pr[0, idx[..., 0], idx[..., 1], :, :]
            pr_sample = torch.sum(pr_sample * interp[:,:,:,:,None,None], axis=3)[0]

        if model.do_radiation:
            soy = date.replace(month=1, day=1)
            toy = int((date - soy).total_seconds()/86400)
            if toy % 3 != 0:
                toy -= toy % 3
            neocache = {}
            neocache, rad = interpget_cached(neocache, "neoradiation_1", toy, date.hour)
            neocache, ang = interpget_cached(neocache, "solarangle_1", toy, date.hour, a=0, b=180/np.pi)
            sa = torch.sin(ang)
            ca = torch.cos(ang)
            arrs = [rad, sa, ca]

            extra = []
            for a in arrs:
                a = a.cuda()
                exa = a[idx[..., 0], idx[..., 1]]
                exa = torch.sum(exa * interp, axis=3)[0]
                del a
                extra.append(exa[:,:,None])

        else:
            extra = []


        sfc = era5["sfc"].to(primary_compute)
        sfc_sample = sfc[0, idx[..., 0], idx[..., 1], :]
        sfc_sample = torch.sum(sfc_sample * interp[:,:,:,:,None], axis=3)[0]
        sera5 = statics["era5"]

        sfc_sample = torch.cat([sfc_sample, sera5] + extra, axis=-1)

        if model.do_pressure:
            pr_sample = pr_sample[:, 1:]

        center_sfc = sfc_sample[:,0]

        sfc_sample = sfc_sample[:, 1:]
        sq = int(np.sqrt(sfc_sample.shape[1]))
        sfc_sample = sfc_sample.permute(0, 2, 1).view(-1, sfc_sample.shape[2], sq, sq)

        static_keys = ["mn30", "mn75"]
        modis_keys = ["modis_"+x for x in static_keys]
        static = {x: statics[x] for x in static_keys + model.do_modis*modis_keys}
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
                #modis = static["modis_"+x][:, 1:].view(-1, sq, sq, 17)
                modis = static["modis_"+x][:, 1:].view(-1, sq, sq)#, 17)
                modis = torch.nn.functional.one_hot(modis.long(), 17)
                static[x] = torch.cat((static[x], modis), dim=3)
            static[x] = static[x].permute(0, 3, 1, 2)
        inp = {}

        if model.do_pressure:
            sq = int(np.sqrt(pr_sample.shape[1]))
            pr_sample = pr_sample.view(-1, sq, sq, 5, 28).permute(0, 3, 4, 1, 2)
            inp["pr"] = pr_sample
        inp["sfc"] = sfc_sample
        for k in static_keys:
            inp[k] = static[k]
        inp["center"] = torch.cat([center[x][:, None] if len(center[x].shape)==1 else center[x] for x in static_keys + model.do_modis * modis_keys], dim=1)
        inp["center"] = torch.cat([inp["center"], center_sfc], dim=1)
        inp = {x: y.half() for x, y in inp.items()}
        return inp

def upload_worker():
    while True:
        task = upload_queue.get()
        if task is None:
            break
        output_fn, h, date = task
        upload_output(output_fn, h, date)
        upload_queue.task_done()

def save_upload_output(out: np.ndarray, h: int, date: str):
    start_time = time.time()
    output_fn = f'bayarea/{h}.npy'
    if not os.path.exists(f'{OUTPUT_PATH.replace("DATESTR", date)}/bayarea'):
        os.makedirs(f'{OUTPUT_PATH.replace("DATESTR", date)}/bayarea')
    np.save(f'{OUTPUT_PATH.replace("DATESTR", date)}/{output_fn}', out)
    print(f"Saved fh {h} ({round((time.time() - start_time) * 1000)} ms)", flush=True)
    gc.collect()
    if S3_UPLOAD:
        upload_queue.put((output_fn, h, date))


def upload_output(output_fn: str, h: int, date: str):
    print(f"Uploading forecast hour {h} to s3", flush=True)
    start_time = time.time()
    s3_fn = f'{S3_OUTPUT_PATH.replace("DATESTR", date)}/{output_fn}'
    S3_CLIENT.upload_file(f'{OUTPUT_PATH.replace("DATESTR", date)}/{output_fn}', BUCKET, s3_fn)
    print(f"Uploaded fh {h} to s3 bucket {BUCKET} at {s3_fn} ({round((time.time() - start_time) * 1000)} ms)", flush=True)


def get_pointy_for_grid(
    datestr: str, 
    frames: list[int], 
    domain: str, 
    resolution: int
) -> None:
    assert resolution == 1, "I don't know what resolution is but I think it's supposed to be 1"
    print("Loading model...")
    load_pointy()
    with open(f'/fast/ignored/hres/{domain}_shapes.pickle', 'rb') as file:
        loaded_data = pickle.load(file)
    lats_shape = loaded_data['lats_shape']
    lons_shape = loaded_data['lons_shape']
    dataset_lats = loaded_data['lats']
    dataset_lons = loaded_data['lons']

    lats, lons = np.meshgrid(dataset_lats, dataset_lons, indexing='ij')
    latlon_grid = np.stack((lats, lons), axis=-1)
    latlon_grid = latlon_grid.reshape(-1, 2)

    vshape = lats_shape, lons_shape

    #NORMALIZATION
    meta = read_metadata()
    levels = meta["levels"]
    n_levels = len(levels)
    with open("/fast/consts/normalization.pickle", "rb") as f:
        normalization = pickle.load(f)
    wanted_indexes = [levels_full.index(x) for x in levels]
    pressure_vars = meta["pressure_vars"]
    # Hardcoded to only use the core sfc vars till we support more
    sfc_vars = [var for var in meta["sfc_vars"]][:4]
    assert len(pressure_vars) == 5, len(pressure_vars)
    assert len(sfc_vars) == 4, len(sfc_vars)
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
    #END NORMALIZATION

    with torch.no_grad():
        print(f"Loading interps for {domain}")
        interps, idxs = pickle.load(open(f"/fast/ignored/hres/{domain}_interps_new.pickle", "rb")) #this is where the grid is created (interp_bayarea.py)
        interps = torch.tensor(interps).unsqueeze(0).to('cuda')
        idxs = torch.tensor(idxs).unsqueeze(0).to('cuda')
        statics = pickle.load(open(f"/fast/ignored/hres/{domain}_statics_new.pickle", "rb"))
        statics = {x: torch.tensor(y).to('cuda') for x, y in statics.items()}
        
        for frame in frames:
            print(f"Processing frame {frame}")

            date = datetime.strptime(datestr, "%Y%m%d%H").replace(tzinfo=timezone.utc) + timedelta(hours=frame)

            # if MODEL_HASH == "WeatherMesh":
            #     path = f"{WM_OUTPUT_PATH}/{datestr}+{frame}.{MODEL_HASH}.npy"
            # else:
            #     path = f"{WM_OUTPUT_PATH}/{datestr}/det/{frame}.{MODEL_HASH}.npy"
            path = f"{WM_OUTPUT_PATH}/{datestr}/det/{frame}.{MODEL_HASH}.npy"
            era5 = np.load(path)
            n_core_vars = len(pressure_vars) * n_levels + len(sfc_vars)
            assert n_core_vars in [144, 129], "expecting 144 or 129 core variables for 28 or 25 levels"
            era5 = era5[:,:,:n_core_vars]

            # normalize
            for i in range(n_core_vars):
                era5[:,:,i] = (era5[:,:,i] - all_means[i]) / np.sqrt(all_stds[i])
            # interp levels if not 28 levels
            if n_levels != 28:
                assert n_levels == 25, "assume either 28 or 25 levels for now"
                # eh no real mesh but this should work
                mesh = SimpleNamespace(n_pr_vars=5)
                era5 = interp_levels(torch.from_numpy(era5), mesh, levels_joank, levels_medium).numpy()
                assert era5.shape == (720, 1440, 144), era5.shape
            # shape is currently (720, 1440, 144). make a dictionary where "sfc" is the first 5 of the third dimension and "pr" is the rest
            sfc_era5 = era5[:,:,-4:]
            pr_era5 = era5[:,:,:-4]
            era5 = {"sfc": sfc_era5, "pr": pr_era5.reshape(720, 1440, 5, 28)}
            
            era5 = {x: torch.tensor(y).unsqueeze(0).to('cuda') for x, y in era5.items()} # keys: pr, sfc 

            domain_points = pickle.load(open(f"/fast/ignored/hres/{domain}_pts.pickle", "rb"))

            full = np.arange(idxs.shape[1]) # full is the number of points in the grid
            outs = [] ; bils = []
            print(f"Number of points in grid {domain}: {len(full)}")
            num_chunks = len(full)//500
            for chunk in np.array_split(full, num_chunks): #splits into chunks of 10
                interpsx = interps[:,chunk]
                idxsx = idxs[:,chunk]
                staticsx = {x: y[chunk] for x, y in statics.items()}
                inpx = build_input(model, era5, interpsx, idxsx, staticsx, date) #pressure
                bils.append(inpx["center"][:,-12:-8]) 
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = model(inpx) #outputs in (length_of_row, num_variables) shape
                outs.append(out.cpu().detach().numpy())
            out = np.concatenate(outs, axis=0)
            out = out.reshape(*vshape, -1)

            normalization['vsby'] = (np.array([10]), np.array([5**2]))
            normalization['skyl1'] = (np.array([2000]), np.array([4000**2]))
            normalization['precip'] = (np.array([10]), np.array([10**2]))

            def n(arr, norm):
                arr = arr.astype(np.float32)
                return arr * np.sqrt(normalization[norm][1][0]) + normalization[norm][0][0]

            tmp = n(out[:,:,0], '167_2t')
            dpt = n(out[:,:,1], '168_2d')
            mslp = n(out[:, :, 2], '151_msl')
            # for old pointy the u and v components are flipped!!
            assert model.n_out == 8, "if this fails we are using old pointy again"
            ucomp = n(out[:,:,3], '165_10u') * 1.94
            vcomp = n(out[:,:,4], '166_10v') * 1.94
            skyl1 = n(out[:,:,5], "skyl1") * 0.3048 # cloud base height in meters
            vsby = n(out[:,:,6], "vsby") * 1.60934 # visibility in km
            precip = n(out[:,:,7], "precip") # precip in mm

            # obtain unique values of the latitudes and longitudes
            lats = np.unique(domain_points[:,:,0])
            lons = np.unique(domain_points[:,:,1])

            #unsqueeze each variable: tmp, dpt, mslp, ucomp, vcomp
            outputs = [tmp, dpt, mslp, ucomp, vcomp, skyl1, vsby, precip]
            for i in range(len(outputs)):
                outputs[i] = np.expand_dims(outputs[i], axis=2)
            output_arr = np.concatenate(outputs, axis=2)

            output_path = f'{OUTPUT_PATH.replace("DATESTR", datestr)}/bayarea'

            save_upload_output(output_arr, frame, datestr)

        meta = {
            'full_varlist': ['temperature', 'dewpoint', 'pressure', 'wind_u', 'wind_v', 'cloud_base_height', 'visibility', 'precipitation'],
            'lats': lats.tolist(),
            'lons': lons.tolist(),
            'res': resolution
        }
        with open(output_path + '/meta.json', 'w') as f:
            json.dump(meta, f)   

def check_processed(date: str, frames: list[int]) -> bool:
    output_path = f'{OUTPUT_PATH.replace("DATESTR", date)}/bayarea'
    processed_count = len(glob.glob(f'{output_path}/*.npy'))
    return processed_count == len(frames)

def get_latest_wm_date(is_old_format: bool) -> datetime:
    latest = None
    if is_old_format:
        for file in os.listdir(WM_OUTPUT_PATH):
            if not file.endswith(".npy"):
                continue
            date = datetime.strptime(file[:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)
            if latest is None or date > latest:
                latest = date
        assert latest is not None, "No WM output found"
        return latest
    else:
        for file in os.listdir(WM_OUTPUT_PATH):
            if not os.path.isdir(os.path.join(WM_OUTPUT_PATH, file)):
                continue
            date = datetime.strptime(file, "%Y%m%d%H").replace(tzinfo=timezone.utc)
            if latest is None or date > latest:
                latest = date
        assert latest is not None, "No WM output found"
        return latest

def get_latest_wm_frames(is_old_format: bool) -> list[int]:
    latest = get_latest_wm_date(is_old_format)
    frames = []
    if is_old_format:
        for file in os.listdir(WM_OUTPUT_PATH):
            if not file.endswith(".npy"):
                continue
            if not file.startswith(latest.strftime("%Y%m%d%H")):
                continue
            frame = int(file.split(".")[0].split("+")[1])
            frames.append(frame)
        return frames
    else:
        for file in os.listdir(os.path.join(WM_OUTPUT_PATH, latest.strftime("%Y%m%d%H"), "det")):
            if not file.endswith(".npy"):
                continue
            frame = int(file.split(".")[0])
            frames.append(frame)
        return frames

def run_and_save(date: Optional[str], domain: str):
    is_old_format = False #True if WM_MODEL == "WeatherMesh" else False
    if date is None:
        date = get_latest_wm_date(is_old_format).strftime("%Y%m%d%H")
        print(f"Latest WM output available is for {date}", flush=True)
    frames = sorted(get_latest_wm_frames(is_old_format))
    if check_processed(date, frames):
        print("Already processed", flush=True)
        return
    get_pointy_for_grid(date, frames, domain, 1)

def run_if_needed(datestr: str, domain: str):
    assert domain == 'bay1', "Expecting to only run for bay1 domain for now"
    load_pointy()
    if S3_UPLOAD:
        upload_thread = threading.Thread(target=upload_worker)
        upload_thread.start()
        run_and_save(datestr, domain)
        upload_queue.put(None)
        upload_thread.join()
    else:
        run_and_save(datestr, domain)

def main():
    print(f"Started process {os.getpid()} for run_rt_pointy", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('date', nargs='?', help='Forecast zero time in YYYYMMDDHH format')
    parser.add_argument('--domain', type=str, default='bay1')
    parser.add_argument('--no-upload', action='store_true')
    parser.add_argument('--wm_name', type=str, default='WeatherMesh')
    parser.add_argument('--wm_variant', type=str, default=DEFAULT_MODEL_HASH)

    args = parser.parse_args()

    if args.no_upload:
        global S3_UPLOAD
        S3_UPLOAD = False
    
    global WM_MODEL
    WM_MODEL = args.wm_name

    global MODEL_HASH
    MODEL_HASH = args.wm_variant
    
    global WM_OUTPUT_PATH, OUTPUT_PATH, S3_OUTPUT_PATH
    WM_OUTPUT_PATH = f'{BASE_OUTPUT_FOLDER}/{WM_MODEL}'
    OUTPUT_PATH = f'{BASE_OUTPUT_FOLDER}/{WM_MODEL}/DATESTR/pointy'
    S3_OUTPUT_PATH = f'{WM_MODEL}/DATESTR/pointy'
    
    run_if_needed(args.date, args.domain)

if __name__ == '__main__':
    main()