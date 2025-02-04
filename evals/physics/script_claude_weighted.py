import dataclasses
import datetime
import functools
import math
import re
from typing import Optional

import cartopy.crs as ccrs
from google.cloud import storage
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
from IPython.display import HTML
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray as xr
import xarray
from IPython.display import HTML
from eval_helpers import *
from plot_helpers import *

Gzarr = xr.open_zarr('gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr')
gc_batch_subset = Gzarr.sel(time=slice('2020-01-01T00:00:00', '2020-01-01T12:00:00'))

gc_batch_subset = analyze_mass_conservation(gc_batch_subset)

# Calculate raw error (unweighted)
raw_error_mean = gc_batch_subset['continuity_error'].sum(dim=['lat', 'lon', 'level']).mean(dim='time')
raw_error_std = gc_batch_subset['continuity_error'].sum(dim=['lat', 'lon', 'level']).std(dim='time')

# Create x-axis in days
days = np.arange(len(raw_error_mean.isel(prediction_timedelta=slice(1, -1)))) / 4

# Create subplot figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot 1: Raw Error
raw_mean = raw_error_mean.isel(prediction_timedelta=slice(1, -1)).values
raw_std = raw_error_std.isel(prediction_timedelta=slice(1, -1)).values

ax1.fill_between(
    days,
    raw_mean - raw_std,
    raw_mean + raw_std,
    color='lightcoral',
    alpha=0.3,
    label='Mean ± 1 Std Dev'
)
ax1.plot(days, raw_mean, color='red', label='Raw Mean')
ax1.set_title('Raw Continuity Error')
ax1.set_ylabel('Continuity Error [kg/m²/s]')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()
ax1.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Plot 2: Volume-Weighted Error

# weight error by volume
gc_error_weighted = gc_batch_subset['continuity_error'] * (gc_batch_subset['volume']/gc_batch_subset['volume'].sum(dim=['lat', 'lon', 'level'])) 
gc_error_weighted_mean = gc_error_weighted.sum(dim=['lat', 'lon', 'level']).mean(dim='time')
gc_error_weighted_std = gc_error_weighted.sum(dim=['lat', 'lon', 'level']).std(dim='time')

weighted_mean = gc_error_weighted_mean.isel(prediction_timedelta=slice(1, -1)).values
weighted_std = gc_error_weighted_std.isel(prediction_timedelta=slice(1, -1)).values

ax2.fill_between(
    days,
    weighted_mean - weighted_std,
    weighted_mean + weighted_std,
    color='lightblue',
    alpha=0.3,
    label='Mean ± 1 Std Dev'
)
ax2.plot(days, weighted_mean, color='blue', label='Volume-Weighted Mean')
ax2.set_title('Volume-Weighted Continuity Error')
ax2.set_xlabel('Forecast Lead Time [days]')
ax2.set_ylabel('Weighted Continuity Error [kg/m²/s]')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()
ax2.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

plt.tight_layout()
plt.savefig('continuity_error_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print the ratio of magnitudes at beginning and end of forecast
start_ratio = abs(weighted_mean[0]/raw_mean[0])
end_ratio = abs(weighted_mean[-1]/raw_mean[-1])
print(f"\nRatio of weighted to raw error:")
print(f"At start of forecast: {start_ratio:.2e}")
print(f"At end of forecast: {end_ratio:.2e}")

# Also let's look at where the errors are occurring vertically
level_mean_error = gc_batch_subset['continuity_error'].mean(dim=['lat', 'lon', 'time'])
level_volume = gc_batch_subset['volume'].mean(dim=['lat', 'lon', 'time'])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(level_mean_error, gc_batch_subset.level, label='Mean Error')
ax.set_ylabel('Level')
ax.set_xlabel('Mean Continuity Error [kg/m²/s]')
ax.set_title('Vertical Distribution of Continuity Error')
ax.grid(True)
ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

# Add volume distribution on secondary axis
ax2 = ax.twiny()
ax2.plot(level_volume, gc_batch_subset.level, 'r--', label='Mean Volume')
ax2.set_xlabel('Mean Volume [m³]', color='r')
ax2.tick_params(axis='x', colors='r')
ax2.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

plt.tight_layout()
plt.savefig('vertical_error_distribution.png', dpi=300, bbox_inches='tight')
plt.show()