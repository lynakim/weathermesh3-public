# %%
from collections import defaultdict
import itertools
import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import TwoSlopeNorm
from netCDF4 import Dataset
import numpy as np
import os
import sys
sys.path.append('/fast/haoxing/deep')
import time
import torch
from tqdm import tqdm
np.set_printoptions(precision=4, suppress=True)

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from PIL import Image
import pickle
import pygrib
import requests
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

from utils import levels_joank, levels_medium, levels_hres, levels_tiny, levels_full, CONSTS_PATH, core_pressure_vars, core_sfc_vars, levels_aurora, levels_ncarhres, get_dates, D, ncar2cloud_names
from data import WeatherDataset, DataConfig, default_collate
import meshes

# %%
# calculate rmse for every variable in every level
bachelor_output_path = '/huge/deep/evaluation/ultralatentbachelor_168M/outputs'
bachelor_error_path = '/huge/deep/evaluation/ultralatentbachelor_168M/errors'

# %%
levels = [500, 700, 850]
pr_errs, sfc_errs = {}, {}
for level in levels:
    for pr_var in core_pressure_vars:
        pr_errs[f"{pr_var}_{level}"] = [[] for _ in range(9)]

for sfc_var in core_sfc_vars:
    sfc_errs[sfc_var] = [[] for _ in range(9)]

for fn in os.listdir(bachelor_error_path):
    if not fn.endswith('.json'): continue
    with open(f"{bachelor_error_path}/{fn}", 'r') as f:
        err = json.load(f)
    fh = err['forecast_dt_hours']
    if fh == 0: continue
    if err['input_time'][:4] != '2020': continue
    rmses = err['rmse']
    for level in levels:
        for pr_var in core_pressure_vars:
            var = f"{pr_var}_{level}"
            pr_errs[var][fh // 24 - 1].append(rmses[var])
    for sfc_var in core_sfc_vars:
        sfc_errs[sfc_var][fh // 24 - 1].append(rmses[sfc_var])

# %%
for key in pr_errs:
    for i in range(9):
        print(f"len(pr_errs[{key}][{i}]): {len(pr_errs[key][i])}")
for key in sfc_errs:
    for i in range(9):
        print(f"len(sfc_errs[{key}][{i}]): {len(sfc_errs[key][i])}")
# %%
# take average
pr_errs_avg, sfc_errs_avg = {}, {}
for key in pr_errs:
    pr_errs_avg[key] = np.zeros(9)
    for i in range(9):
        vals = sorted(pr_errs[key][i])[3:-3]
        pr_errs_avg[key][i] = np.mean(vals)
for key in sfc_errs:
    sfc_errs_avg[key] = np.zeros(9)
    for i in range(9):
        vals = sorted(sfc_errs[key][i])[3:-3]
        sfc_errs_avg[key][i] = np.mean(vals)

# %%
hres_perf_path = '/fast/weatherbench2_results/deterministic/hres_vs_analysis_2020_deterministic.nc'
hres = xr.open_dataset(hres_perf_path)
hres_dt_idxs = np.array([4*i for i in range(1, 10)])
# %%
hres_pr_errs, hres_sfc_errs = {}, {}
for level in levels:
    for pr_var in core_pressure_vars:
        var = f"{pr_var}_{level}"
        cv = ncar2cloud_names[pr_var]
        sel = {'region': 'global', 'metric': 'rmse', 'level': level}
        hres_rmses = hres[cv].sel(**sel).values
        if pr_var == "133_q":
            hres_rmses *= 1e6
        assert len(hres_rmses) == 41
        hres_pr_errs[var] = hres_rmses[hres_dt_idxs]
for sfc_var in core_sfc_vars:
    cv = ncar2cloud_names[sfc_var]
    sel = {'region': 'global', 'metric': 'rmse'}
    hres_rmses = hres[cv].sel(**sel).values
    assert len(hres_rmses) == 41
    hres_sfc_errs[sfc_var] = hres_rmses[hres_dt_idxs]

# %%
pr_pct_improvements, sfc_pct_improvements = {}, {}
for key in pr_errs_avg:
    pr_pct_improvements[key] = (pr_errs_avg[key] - hres_pr_errs[key]) / hres_pr_errs[key] * 100
for key in sfc_errs_avg:
    sfc_pct_improvements[key] = (sfc_errs_avg[key] - hres_sfc_errs[key]) / hres_sfc_errs[key] * 100

# %%
# restructure into numpy arrays for each pr var
data_u = np.zeros((len(levels), 9))
data_v = np.zeros((len(levels), 9))
data_t = np.zeros((len(levels), 9))
data_q = np.zeros((len(levels), 9))
data_z = np.zeros((len(levels), 9))
for key in pr_errs_avg:
    _, var, level = key.split('_')
    level = int(level)
    if var == 'u':
        data_u[levels.index(level)] = pr_pct_improvements[key]
    elif var == 'v':
        data_v[levels.index(level)] = pr_pct_improvements[key]
    elif var == 'q':
        data_q[levels.index(level)] = pr_pct_improvements[key]
    elif var == 't':
        data_t[levels.index(level)] = pr_pct_improvements[key]
    elif var == 'z':
        data_z[levels.index(level)] = pr_pct_improvements[key]
# %%

pr_variables = [
    (data_u, "U"),
    (data_v, "V"),
    (data_t, "T"),
    (data_q, "Q"),
    (data_z, "Z")
]

surface_variables = [
    (sfc_pct_improvements["165_10u"][None,], "10U"),
    (sfc_pct_improvements["166_10v"][None,], "10V"),
    (sfc_pct_improvements["151_msl"][None,], "MSL"),
    (sfc_pct_improvements["167_2t"][None,], "2T")
]

lead_times = np.arange(1, 10)

# %%
# get the min out of all numbers
min_err = 100
for i in range(9):
    for j in range(len(levels)):
        for data, _ in pr_variables:
            min_err = min(min_err, data[j, i])
        for data, _ in surface_variables:
            min_err = min(min_err, data[0, i])

# %%
#plt.style.use('/fast/haoxing/deep/wb.mplstyle')
plt.style.use('default')
fig = plt.figure(figsize=(15, 3), dpi=400, constrained_layout=True)
fig.patch.set_alpha(0)
gs = fig.add_gridspec(2, len(pr_variables), height_ratios=[3, 1])
norm = TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)

# Main variables (first row)
axes_main = [fig.add_subplot(gs[0, i]) for i in range(len(pr_variables))]
for ax, (data, label) in zip(axes_main, pr_variables):
    ax.grid(False)
    im = ax.imshow(data, aspect='auto', cmap='bwr', norm=norm, origin='lower',
                   extent=[lead_times[0], lead_times[-1], levels[-1], levels[0]])
    ax.set_title(label, fontsize=16)
    ax.set_xticks(np.arange(0.5,9.5,1) * 8/9 + 1)
    ax.set_xticklabels(lead_times)
    ax.set_yticks(np.arange(0.5,3.5,1) * 350/3 + 500)
    ax.tick_params(labelsize=12)
    if ax == axes_main[0]:
        ax.set_yticklabels(levels)
        ax.set_ylabel("Pressure / hPa", fontsize=14)
    else:
        ax.set_yticklabels(["  "] * 3)

# Surface variables (single row below)
axes_surface = [fig.add_subplot(gs[1, i]) for i in range(len(surface_variables))]
for ax, (data, label) in zip(axes_surface, surface_variables):
    ax.grid(False)
    im = ax.imshow(data, aspect='auto', cmap='bwr', norm=norm, origin='lower',
                   extent=[lead_times[0], lead_times[-1], 0, 1])  # Dummy vertical extent
    ax.set_title(label, fontsize=16)
    ax.set_xticks(np.arange(0.5,9.5,1) * 8/9 + 1)
    ax.set_xticklabels(lead_times)
    ax.set_yticks([])  # No vertical ticks for single-layer data
    ax.tick_params(labelsize=12)
    ax.set_xlabel("Lead time / days", fontsize=14)

# Add a single colorbar for all plots
cbar = fig.colorbar(im, ax=axes_main + axes_surface, orientation='vertical', pad=0.02, aspect=10)
cbar.set_label("RMSE relative to IFS HRES", fontsize=14)
cbar.ax.set_yticklabels(["-30%", "-20%", "-10%", "0%", "10%", "20%", "30%"])
cbar.ax.tick_params(labelsize=12)

#plt.suptitle("Scorecard comparing WeatherMesh vs HRES", fontsize=14, fontweight='bold')

# save
#plt.savefig('/fast/haoxing/deep/blog/wm_v2/scorecard.png')
plt.savefig('/fast/haoxing/scorecard_hres.png')

# %%
