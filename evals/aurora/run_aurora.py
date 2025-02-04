# %%
from datetime import datetime, timezone, timedelta
from einops import rearrange
import numpy as np
import pickle
import pygrib
import sys
sys.path.append("/fast/haoxing/deep")
import time
import torch
import matplotlib.pyplot as plt

from utils import levels_aurora, levels_joank, levels_medium, levels_full, levels_hres, levels_tiny

from aurora import Batch, Metadata, Aurora, rollout

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Set to the GPU index you want, e.g., "0" or "0,1"

core_pressure_vars = ["129_z", "130_t", "131_u", "132_v", "133_q"]
core_sfc_vars = ["165_10u", "166_10v", "167_2t", "151_msl"]
ordered_pr_vars = ["z", "t", "u", "v", "q"]
ordered_sfc_vars = ["10u", "10v", "2t", "msl"]

with open ("/fast/haoxing/aurora-0.25-static-hrest0.pickle", "rb") as f:
    aurora_statics = pickle.load(f)
with open("/fast/consts/normalization.pickle", "rb") as f:
    normalization = pickle.load(f)

VAR_NAMES = {
    "Geopotential Height": "129_z",
    "Temperature": "130_t",
    "U component of wind": "131_u",
    "V component of wind": "132_v",
    "Specific humidity": "133_q"
}

SURFACE_VARS = {
    "10 metre U wind component": "165_10u",
    "10 metre V wind component": "166_10v",
    "2 metre temperature": "167_2t",
    "Mean sea level pressure": "151_msl"
}

OUTPUT_DIR = "/huge/users/haoxing/aurora/finetuned5"

# %%
def load_hres_wx_archive(date: datetime):
    date_str = date.strftime("%Y%m%d")
    stream = "oper" if date.hour == 0 else "scda"
    wx_archive_hres_grbs = f"/huge/proc/weather-archive/ecmwf/{date_str}/{date.hour:02d}z/ifs/0p25/{stream}/{date_str}{date.hour:02d}0000-0h-{stream}-fc.grib2"
    grbs = pygrib.open(wx_archive_hres_grbs)
    prdic = {v: [] for k, v in VAR_NAMES.items()}
    whlev = {v: [] for k, v in VAR_NAMES.items()}
    sfcdic = {v: [] for k, v in SURFACE_VARS.items()}

    for grb in grbs:
        name = grb.name
        name = name.replace("height", "Height")

        assert levels_tiny == levels_aurora
        if name in VAR_NAMES and grb.typeOfLevel == "isobaricInhPa" and grb.level in levels_tiny:
            scaling = 1
            if name == "Geopotential Height":
                scaling = 9.80665

            prdic[VAR_NAMES[name]].append((grb.level, grb.values[:].copy().astype(np.float32) * scaling))
            whlev[VAR_NAMES[name]].append(int(grb.level))

        if name in SURFACE_VARS:
            sfcdic[SURFACE_VARS[name]].append(grb.values[:].copy().astype(np.float32))

    pr = []
    sfc = []
    for v in core_pressure_vars:
        sorted_levels = sorted(whlev[v])
        # sort data by level
        prdic[v].sort(key=lambda x: x[0])

        if sorted_levels != levels_tiny:
            print("uhh")
            for a, b in zip(sorted_levels, levels_tiny):
                print(a, b)
        assert sorted_levels == levels_tiny
        values = [x[1] for x in prdic[v]]
        x = np.stack(values).transpose(1, 2, 0)
        #print(v, np.mean(x), np.std(x), np.max(np.abs(x)))
        pr.append(x)

    for v in core_sfc_vars:
        x = np.array(sfcdic[v])
        #print(f"{v}: mean={round(np.mean(x), 3)} std={round(np.std(x), 3)}")
        sfc.append(x[0])

    sfc_data = np.array(sfc).transpose(1, 2, 0)
    pr_data = np.array(pr).transpose(1, 2, 0, 3)
    sfc_data = np.roll(sfc_data, 720, axis=1)
    pr_data = np.roll(pr_data, 720, axis=1)
    return sfc_data, pr_data


# %%
def make_aurora_batch(ts: int, mode: str = "era5"):
    """Formats ERA5 or HRES t0 into the Aurora batch format."""
    assert mode in ("era5", "hres_rt", "hres_wx")

    prev_ts = ts - 3600 * 6
    curr, prev = datetime.utcfromtimestamp(ts), datetime.utcfromtimestamp(prev_ts)
    if mode == "era5":
        # pr: 720, 1440, 5, 28
        curr_npz = np.load(f"/fast/proc/era5/f000/{curr.year}{curr.month:02}/{ts}.npz")
        prev_npz = np.load(f"/fast/proc/era5/f000/{prev.year}{prev.month:02}/{prev_ts}.npz")
        levels = levels_medium
    elif mode == "hres_rt":
        # pr: 721, 1440, 5, 20
        curr_npz = np.load(f"/fast/proc/hres_rt/f000/{curr.strftime('%Y%m%d%H')}.npz")
        prev_npz = np.load(f"/fast/proc/hres_rt/f000/{prev.strftime('%Y%m%d%H')}.npz")
        levels = levels_hres
    
    if mode in ["era5", "hres_rt"]:
        # unnormalize era5
        used_idxs = [levels_full.index(x) for x in levels]
        sfc_means, sfc_stds = [], []
        for var in core_sfc_vars:
            mean, std = normalization[var]
            sfc_means.append(mean)
            sfc_stds.append(std)
        n = len(used_idxs)
        pr_means, pr_stds = np.zeros((5, n)), np.zeros((5, n))
        for i, var in enumerate(core_pressure_vars):
            mean, std = normalization[var]
            pr_means[i] = mean[used_idxs]
            pr_stds[i] = std[used_idxs]

        sfc2 = curr_npz["sfc"] * np.sqrt(np.concatenate(sfc_stds)) + np.concatenate(sfc_means)
        sfc1 = prev_npz["sfc"] * np.sqrt(np.concatenate(sfc_stds)) + np.concatenate(sfc_means)
        pr2 = curr_npz["pr"] * np.sqrt(pr_stds) + pr_means
        pr1 = prev_npz["pr"] * np.sqrt(pr_stds) + pr_means

    if mode == "hres_wx":
        sfc1, pr1 = load_hres_wx_archive(prev)
        sfc2, pr2 = load_hres_wx_archive(curr)
        assert sfc1.shape == (721, 1440, 4)
        assert pr1.shape == (721, 1440, 5, 13)
        levels = levels_aurora
    
    sfc1 = rearrange(sfc1, "lat lon sfc -> sfc lat lon")
    sfc2 = rearrange(sfc2, "lat lon sfc -> sfc lat lon")
    pr1 = rearrange(pr1, "lat lon pr lev -> pr lev lat lon")
    pr2 = rearrange(pr2, "lat lon pr lev -> pr lev lat lon")

    if mode == "era5":
        # pad for 721st lat
        sfc1 = np.pad(sfc1, ((0, 0), (0, 1), (0, 0)), mode="edge")
        sfc2 = np.pad(sfc2, ((0, 0), (0, 1), (0, 0)), mode="edge")
        pr1 = np.pad(pr1, ((0, 0), (0, 0), (0, 1), (0, 0)), mode="edge")
        pr2 = np.pad(pr2, ((0, 0), (0, 0), (0, 1), (0, 0)), mode="edge") 

    aurora_idxs = [levels.index(l) for l in levels_aurora]
    surf_vars = {k: torch.from_numpy(np.stack([sfc1[i], sfc2[i]], axis=0)[None])
                        for i, k in enumerate(ordered_sfc_vars)}
    atmos_vars = {k: torch.from_numpy(np.stack([pr1[i, aurora_idxs], pr2[i, aurora_idxs]], axis=0)[None])
                        for i, k in enumerate(ordered_pr_vars)}

    batch = Batch(
        surf_vars=surf_vars,
        #static_vars={k: torch.from_numpy(aurora_statics[k][:-1,:]) for k in ("lsm", "z", "slt")},
        static_vars={k: torch.from_numpy(aurora_statics[k]) for k in ("lsm", "z", "slt")},
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=torch.linspace(90, -90, 721),
            lon=torch.linspace(0, 360, 1441)[:-1],
            time=(curr,), # <- um unsure if this matters; update 12/11/2024: it sure does past haoxing is an idiot
            atmos_levels=levels_aurora,
        ),
    )

    return batch

# %%
# from aurora import AuroraSmall

# model = AuroraSmall()
# model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
# model.eval()
# model = model.to("cuda")

# # %%
# model = Aurora(use_lora=False)  # The pretrained version does not use LoRA.
# model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
# model.eval()
# model = model.to("cuda")

# %%
model = Aurora()
model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
model.eval()
model = model.to("cuda")
#model = model.to("cuda:0")
#model = model.to("cuda:1")
# %%
def save_aurora_output(pred: Batch, forecast_init_ts: int, lead_time: int):
    sfc_vars = {k: v.squeeze().cpu().numpy() for k, v in pred.surf_vars.items()}
    atmos_vars = {k: v.squeeze().cpu().numpy() for k, v in pred.atmos_vars.items()}
    forecast_init = datetime.utcfromtimestamp(forecast_init_ts)
    time_str = forecast_init.strftime("%Y%m%d%H")
    os.makedirs(f"{OUTPUT_DIR}/{time_str}", exist_ok=True)
    np.savez(f"{OUTPUT_DIR}/{time_str}/{lead_time}.npz", sfc=sfc_vars, atmos=atmos_vars)
    print(f"Saved {time_str}/{lead_time}.npz")

# %%

def run_aurora(start_date: datetime, end_date: datetime, days: int = 14):
    current = start_date
    forecast_hours = 6
    num_steps = int((24/forecast_hours) * days)

    while current <= end_date:
        t0 = time.time()
        ts = int(current.timestamp())
        mode = "hres_wx"
        batch = make_aurora_batch(ts, mode=mode)
        #batch_rt = make_aurora_batch(ts, mode="hres_rt")
        print(f"making batch in mode {mode}")
        #print(f"batch.surf_vars['2t'].shape: {batch.surf_vars['2t'].shape}")
        with torch.inference_mode():
            print(f"Rolling out {current}")
            preds = [pred.to("cpu") for pred in rollout(model, batch, steps=num_steps)]
            # ugh the forecast hour here is off by 6 hours
            for i, pred in enumerate(preds):
                save_aurora_output(pred, ts, i * forecast_hours)
            try:
                # print 10-day RMSE for quick check
                hours = 240
                pred1 = preds[hours // 6 - 1]
                pred_ts = ts + hours * 3600
                era5_t = np.load(f"/fast/proc/era5/f000/2024{datetime.utcfromtimestamp(pred_ts).month:02}/{pred_ts}.npz")
                hres_t = np.load(f"/fast/proc/hres_rt/f000/{datetime.utcfromtimestamp(pred_ts).strftime('%Y%m%d%H')}.npz")
                era5_sfc_t = era5_t["sfc"]
                hres_sfc_t = hres_t["sfc"]
                era5_target_2t = era5_sfc_t[:720,:,2] * np.sqrt(236.8513564) + 287.48915052
                hres_target_2t = hres_sfc_t[:720,:,2] * np.sqrt(236.8513564) + 287.48915052
                era5_target_z500 = (era5_t["pr"][:720,:,0,14] * np.sqrt(normalization["129_z"][1][21]) + normalization["129_z"][0][21]).astype(np.float32)
                hres_target_z500 = hres_t["pr"][:720,:,0,11] * np.sqrt(normalization["129_z"][1][21]) + normalization["129_z"][0][21].astype(np.float32)
                lats = np.linspace(90, -90, 720)
                weights = np.cos(lats * np.pi/180)
                weights = np.stack([weights] * 1440, axis=1)
                era5_2t_rmse = np.sqrt(np.average((pred1.surf_vars["2t"].squeeze().numpy() - era5_target_2t) ** 2, axis=(0, 1), weights=weights))
                hres_2t_rmse = np.sqrt(np.average((pred1.surf_vars["2t"].squeeze().numpy() - hres_target_2t) ** 2, axis=(0, 1), weights=weights))
                era5_z500_rmse = np.sqrt(np.average((pred1.atmos_vars["z"].squeeze().numpy()[7] - era5_target_z500) ** 2, axis=(0, 1), weights=weights))
                hres_z500_rmse = np.sqrt(np.average((pred1.atmos_vars["z"].squeeze().numpy()[7] - hres_target_z500) ** 2, axis=(0, 1), weights=weights))
                print(f"Target: ERA-5")
                print(f"{hours}h 2T RMSE: {era5_2t_rmse}, z500 RMSE: {era5_z500_rmse}")
                print(f"Target: HRES")
                print(f"{hours}h 2T RMSE: {hres_2t_rmse}, z500 RMSE: {hres_z500_rmse}")
            except Exception as e:
                print(f"Failed to calculate RMSE: {e}")
                
        current += timedelta(hours=24)
        print(f"Finished {current}, took {time.time() - t0:.2f}s")

# %%
# batch_hres = make_aurora_batch(ts, mode="hres")
# # %%
# # pred1 = preds[0]
# # pred1
# # # %%
# # era5_t = np.load("/fast/proc/era5/f000/202403/1710028800.npz")
# # # (720, 1440, 5, 28) and (720, 1440, 4)
# # # %%
# # sfc_t = era5_t["sfc"]
# # sfc_t.shape
# # # %%
# # target_2t = sfc_t[:,:,2] * np.sqrt(236.8513564) + 287.48915052
# # # %%
# # np.sqrt(np.mean((pred1.surf_vars["2t"].squeeze().numpy() - target_2t) ** 2))
# # # %%
# # plt.imshow(pred1.surf_vars["2t"].squeeze().numpy())
# # # %%
# # plt.imshow(target_2t)
# # %%
# batch = Batch(
#     surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
#     static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
#     atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
#     metadata=Metadata(
#         lat=torch.linspace(90, -90, 17),
#         lon=torch.linspace(0, 360, 32 + 1)[:-1],
#         time=(datetime(2020, 6, 1, 12, 0),),
#         atmos_levels=(100, 250, 500, 850),
#     ),
# )
# %%

# %%
if __name__ == "__main__":
    try:
        start_date_str = sys.argv[1]
        end_date_str = sys.argv[2]
    except:
        print("usage: python3 evals/aurora/run_aurora.py <start_date> <end_date>")
        print("e.g. python3 evals/aurora/run_aurora.py 20240101 20240131")
        sys.exit(1)
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    start_date = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_date = datetime.strptime(end_date_str, "%Y%m%d")
    end_date = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc)
    run_aurora(start_date, end_date)