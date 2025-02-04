# %%
from datetime import datetime, timezone
import argparse
import sys
import gc
import os
import time
import pickle
import numpy as np
import torch

# %%
# OLD
# from utils import CONSTS_PATH
# from eval import unnorm_output
# from evals import to_filename
# from evals.evaluate_error import save_instance
# from evals.package_neo import get_neoquadripede, get_hegelcasioquad, get_hegelcasiopony

# %%
# Use gen1
from gen1.utils import (
    CONSTS_PATH,
    unnorm_output,
    to_filename,
    save_instance,
)
from gen1.package import get_neoquadripede
# %%

def save_upload_output(out, h, date, output_path):
    output_fn = save_output(out, h, date, output_path)
    # upload_queue.put((output_fn, h))

def save_output(out, h, date, output_path):
    print(f"Saving forecast hour {h}", flush=True)
    start_time = time.time()
    # Save output filename (including tags added by save_instance)

    output_fn = save_instance(out, f'{output_path}/{to_filename(date, round(h), always_plus=True)}', model.config.inputs[0])[len(output_path)+1:]
    print(f"Saved fh {h} ({round((time.time() - start_time) * 1000)} ms)", flush=True)
    gc.collect()
    return output_fn

# %%
# normalize the input
def normalize_input(x: np.ndarray, mesh) -> np.ndarray:
    start_shape = x.shape
    with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
        norm = pickle.load(f)

    pressure_vars = mesh.pressure_vars
    sfc_vars = mesh.sfc_vars

    # shapes
    # (720, 1440, 5, 28) (720, 1440, 4)
    x = x.squeeze()
    x_pr, x_sfc = x[:,:,:140], x[:,:,140:]
    x_pr = np.reshape(x_pr, (720, 1440, 5, 28))
    print("shapes (x_pr, x_sfc): ", x_pr.shape, x_sfc.shape)

    mean_sfc_arr, std_sfc_arr = np.empty_like(x_sfc), np.empty_like(x_sfc)
    for i, var in enumerate(sfc_vars):
        mean, norm_var = norm[var]
        mean_sfc_arr[:,:,i] = mean # shape (720, 1440, 4)
        std_sfc_arr[:,:,i] = np.sqrt(norm_var) # shape (720, 1440, 4)
    x_sfc_normed = (x_sfc - mean_sfc_arr) / std_sfc_arr

    wh_lev = mesh.wh_lev
    mean_pr_arr, std_pr_arr = np.empty_like(x_pr), np.empty_like(x_pr) # (720, 1440, 5, 28)
    for i, var in enumerate(pressure_vars):
        mean, norm_var = norm[var]
        mean_pr_arr[:,:,i,:] = mean[wh_lev] # shape (720, 1440, 5, 28)
        std_pr_arr[:,:,i,:] = np.sqrt(norm_var)[wh_lev] # shape (720, 1440, 5, 28)
    x_pr_normed = (x_pr - mean_pr_arr) / std_pr_arr

    # check magnitude now
    print("Norms (sfc, pr): ", np.abs(x_sfc_normed).mean(), np.abs(x_pr_normed).mean())

    x = np.concatenate([x_pr_normed.reshape(720, 1440, -1), x_sfc_normed], axis=-1)
    assert x.shape == start_shape
    return x

def process_input(file_path: str, mesh):
    x = np.load(file_path)
    print(f"np.abs(x).mean(): {np.abs(x).mean()}")
    x = normalize_input(x, mesh)
    x = torch.from_numpy(x).unsqueeze(0)
    # it appears the model expects a list of two tensors,
    # with the first being the IC and the second being a 
    # (1,) tensor of the timestamp
    x_ts = [x, torch.tensor([date.timestamp()])]
    return x_ts

def run_model(model, x_ts: list[torch.Tensor], time_horizon: int, output_path: str):
    def save_hour(ddt, y):
        y = y.to('cuda')
        xu, y = unnorm_output(x_ts[0], y, model, ddt, y_is_deltas=False, skip_x=True, pad_x=True)

        h = round(ddt)
        save_upload_output(y.to("cpu").numpy(), h, date, output_path)

    with torch.no_grad():
        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            x_ts = [xx.to('cuda') for xx in x_ts]
            model.rollout(
                x_ts,
                time_horizon=time_horizon,
                min_dt=6,
                dt_dict={120: 6},
                callback=save_hour,
        )
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="neoquad")
    parser.add_argument("--date", type=str, default="2024072518")
    parser.add_argument("--input_path", type=str, default="/fast/wbhaoxing/DAICs/ens0/0.E3rN.npy")
    parser.add_argument("--output_path", type=str, default="/fast/realtime/outputs/WeatherMeshDA_gen1")
    parser.add_argument("--time_horizon", type=int, default=168)
    args = parser.parse_args()
    
    assert args.model == "neoquad", "this script only supports neoquad right now"
    assert len(args.date) == 10, "date must be in the format YYYYMMDDHH"
    # neoquad is the 6-hour step model without adapter
    device = torch.device("cuda:0")
    model = get_neoquadripede().to(device)
    model.eval()
    date = datetime.strptime(args.date, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    x_ts = process_input(args.input_path, model.config.inputs[0])
    run_model(model, x_ts, args.time_horizon, args.output_path)
    

# %%
# hegelcasioquad - try to run the same model as run_rt_det.py first

# model = get_hegelcasioquad()
# device = torch.device("cuda:0")
# model = model.to(device)
# # %%
# date = "2024070818"
# date = datetime.strptime(date, "%Y%m%d%H").replace(tzinfo=timezone.utc)
# model.config.inputs[1].source = "hres_rt-13"
# model.config.inputs[1].input_levels = levels_hres
# model.config.inputs[1].intermediate_levels = [levels_tiny]

# dataset = NeoWeatherDataset(
#     NeoDataConfig(
#         inputs=model.config.inputs,
#         outputs=model.config.inputs,  # the output format. Set ot be identical to the input format
#         timesteps=[0],  # timesteps to load for output eval, set to 0 to make this happy
#         requested_dates=[date],
#         use_mmap=False,
#         only_at_z=[0,6,12,18],  # The hours for each date
#         clamp_output=np.inf,  # Do not bound the output; this is only used in training
#         realtime=True,
#     )
# )

# dataset.check_for_dates()

# sample = default_collate([dataset[0]])
# x = sample[0]

# # %%
# with torch.no_grad():
#     with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
#         x = [xx.to('cuda') for xx in x]
#         model.rollout(
#             x,
#             time_horizon=510,
#             min_dt=1,
#             dt_dict={120: 6},
#             callback=None,
#             ic_callback=None,
#     )
# %%
# da_out = np.load("/fast/realtime/outputs/WeatherMeshDA/2024072518+6.E3rN.npy")
# wm_out = np.load("/fast/realtime/outputs/WeatherMesh/2024070818+6.MA1l.npy")
# # %%
# da_out.shape
# # %%
# da_out[0,0,0]
# # %%
# ens0_out = np.load("/fast/realtime/outputs/WeatherMeshDA/ens0/2024072518+6.E3rN.npy")
# ens1_out = np.load("/fast/realtime/outputs/WeatherMeshDA/ens1/2024072518+6.E3rN.npy")
# # %%
# from evals.package_neo import get_neocasio, get_neohegel, get_shallowpony
# device = torch.device("cuda:0")
# model = get_shallowpony().to(device)
# model.eval()
# # %%
# # save state dict
# torch.save(model.state_dict(), "/fast/evaluation/shallowpony_374M/weights/state_dict_epoch4_iter479992_loss0.088.pt")
# %%