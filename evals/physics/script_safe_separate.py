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

# Calculate means and standard deviations for each component
# Divergence
div_mean = gc_batch_subset['mass_flux_divergence'].sum(dim=['lat', 'lon', 'level']).mean(dim='time')
div_std = gc_batch_subset['mass_flux_divergence'].sum(dim=['lat', 'lon', 'level']).std(dim='time')

# Tendency
tend_mean = gc_batch_subset['air_density_tendency'].sum(dim=['lat', 'lon', 'level']).mean(dim='time')
tend_std = gc_batch_subset['air_density_tendency'].sum(dim=['lat', 'lon', 'level']).std(dim='time')

# Continuity Error
error_mean = gc_batch_subset['continuity_error'].sum(dim=['lat', 'lon', 'level']).mean(dim='time')
error_std = gc_batch_subset['continuity_error'].sum(dim=['lat', 'lon', 'level']).std(dim='time')

# Create x-axis in days (4 timesteps per day since data is 6-hourly)
days = np.arange(len(error_mean.isel(prediction_timedelta=slice(1, -1)))) / 4

# Create figure
plt.figure(figsize=(12, 6))

# Plot all components with filled envelopes
slice_idx = slice(1, -1)  # For cleaner code

# Divergence
plt.plot(days, div_mean.isel(prediction_timedelta=slice_idx), 
         color='blue', label='Mass Flux Divergence', linewidth=2)
plt.fill_between(days, 
                 div_mean.isel(prediction_timedelta=slice_idx) - div_std.isel(prediction_timedelta=slice_idx),
                 div_mean.isel(prediction_timedelta=slice_idx) + div_std.isel(prediction_timedelta=slice_idx),
                 color='blue', alpha=0.2)

# Tendency
plt.plot(days, tend_mean.isel(prediction_timedelta=slice_idx), 
         color='green', label='Density Tendency', linewidth=2)
plt.fill_between(days,
                 tend_mean.isel(prediction_timedelta=slice_idx) - tend_std.isel(prediction_timedelta=slice_idx),
                 tend_mean.isel(prediction_timedelta=slice_idx) + tend_std.isel(prediction_timedelta=slice_idx),
                 color='green', alpha=0.2)

# Continuity Error
plt.plot(days, error_mean.isel(prediction_timedelta=slice_idx), 
         color='red', label='Continuity Residual', linewidth=2)
plt.fill_between(days,
                 error_mean.isel(prediction_timedelta=slice_idx) - error_std.isel(prediction_timedelta=slice_idx),
                 error_mean.isel(prediction_timedelta=slice_idx) + error_std.isel(prediction_timedelta=slice_idx),
                 color='red', alpha=0.2)

plt.title('Mass Conservation Components Over Forecast Time')
plt.xlabel('Forecast Lead Time [days]')
plt.ylabel('Rate of Change [kg/m²/s]')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Use scientific notation for y-axis
plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Save image with higher DPI for better quality
plt.savefig('mass_conservation_combined.png', dpi=300, bbox_inches='tight')
plt.show()