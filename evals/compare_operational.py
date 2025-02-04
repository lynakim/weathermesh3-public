# %%
import json
import torch
import numpy as np
import sys
sys.path.append('../')
import datetime
from collections import defaultdict
from eval import eval_metric
from model_latlon.decoder import gather_geospatial_weights
import meshes

date = '2025010600'
wm_meta = json.load(open(f"/fast/realtime/outputs/WeatherMesh/meta.Qfiz.json"))
wmb_meta = json.load(open("/fast/realtime/outputs/WeatherMesh-Beta/meta.I4gx.json"))
assert wm_meta['res'] == 0.25 and wmb_meta['res'] == 0.25, "res must be 0.25"

grid = meshes.LatLonGrid()
gsweights=gather_geospatial_weights(grid)
    
era5t_var_mapping = {
 '129_z_500': 'geopotential_500',
 '129_z_850': 'geopotential_850',
 '130_t_500': 't_500',
 '130_t_850': 't_850',
 '131_u_500': 'u_500',
 '131_u_850': 'u_850',
 '132_v_500': 'v_500',
 '132_v_850': 'v_850',
 '133_q_500': 'q_500',
 '133_q_850': 'q_850',
 '165_10u': 'u10m',
 '166_10v': 'v10m',
 '167_2t': 't2m',
 '151_msl': 'mslp',
}

wm_rmses = defaultdict(list)
wmb_rmses = defaultdict(list)
hours = list(range(0,24*2)) + list(range(24*2, 24*5,6))

for h in hours:
    wm = torch.from_numpy(np.load(f"/fast/realtime/outputs/WeatherMesh/{date}/det/{h}.Qfiz.npy"))
    wmb = torch.from_numpy(np.load(f"/fast/realtime/outputs/WeatherMesh-Beta/{date}/det/{h}.I4gx.npy"))
    era5t_date = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), int(date[8:10])) + datetime.timedelta(hours=h)
    print(f'{h}h RMSEs')
    for k, v in era5t_var_mapping.items():
        era5t = torch.from_numpy(np.load(f"/huge/proc/era5t/{era5t_date.strftime('%Y%m%d%H')}/{v}.npy")[:720, :])
        wm_var = wm[:, :, wm_meta['full_varlist'].index(k)]
        wmb_var = wmb[:, :, wmb_meta['full_varlist'].index(k)]
        wm_rmse = eval_metric(wm_var, era5t, gsweights, grid, stdout=False)
        wmb_rmse = eval_metric(wmb_var, era5t, gsweights, grid, stdout=False)
        print(f'{k}: WM {float(wm_rmse[0][0])}, WM-Beta {float(wmb_rmse[0][0])}')
        wm_rmses[k].append(float(wm_rmse[0][0]))
        wmb_rmses[k].append(float(wmb_rmse[0][0]))
    
# %%
import matplotlib.pyplot as plt
# plot the rmse of 14 variables in separate plots
fig, axs = plt.subplots(4,4, figsize=(12,12))
for i, k in enumerate(list(wm_rmses.keys())):
    axs[i//4, i%4].plot(hours, wm_rmses[k], label=f'WM')
    axs[i//4, i%4].plot(hours, wmb_rmses[k], label=f'WM-Beta')
    # x axis label
    axs[i//4, i%4].set_xlabel('Hour')
    axs[i//4, i%4].legend()
    axs[i//4, i%4].set_title(k)
plt.suptitle(f'RMSE of WM and WM-Beta vs ERA5 upto 5d (init {date})')
plt.tight_layout()
plt.show()
# %%
from matplotlib import colors
fig, axs = plt.subplots(2,3, figsize=(24,8))
datasets = [era5t, wm_var]
norm = colors.Normalize(vmin=np.min(datasets), vmax=np.max(datasets))

axs[0,0].imshow(era5t, norm=norm)
axs[0,1].imshow(wm_var, norm=norm)
axs[1,0].imshow(era5t, norm=norm)
axs[1,1].imshow(wmb_var, norm=norm)
err1 = era5t-wm_var
err2 = era5t-wmb_var
datasets = [err1, err2]
norm = colors.Normalize(vmin=np.min(datasets), vmax=np.max(datasets))
axs[0,2].imshow(err1, norm=norm)
axs[1,2].imshow(err2, norm=norm)

# %%

