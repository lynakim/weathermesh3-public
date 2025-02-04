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
gc_batch_subset = Gzarr.sel(time=slice('2020-01-01T00:00:00', '2020-01-07T00:00:00'))

gc_batch_subset = analyze_mass_conservation(gc_batch_subset)

gc_mean = gc_batch_subset['continuity_error'].sum(dim=['lat', 'lon', 'level']).mean(dim='time')
gc_std = gc_batch_subset['continuity_error'].sum(dim=['lat', 'lon', 'level']).std(dim='time')
# Create x-axis in days (4 timesteps per day since data is 6-hourly)
days = np.arange(len(gc_mean.isel(prediction_timedelta=slice(1, -1)))) / 4

# Create the plot
plt.figure(figsize=(12, 6))

# Mean and envelope boundaries (mean ± 1 standard deviation)
mean_values = gc_mean.isel(prediction_timedelta=slice(1, -1)).values
std_values = gc_std.isel(prediction_timedelta=slice(1, -1)).values

# Create shaded envelope
plt.fill_between(
    days,
    mean_values - std_values,  # Lower bound
    mean_values + std_values,  # Upper bound
    color='lightblue',  # Adjust color to preference
    alpha=0.3,  # Transparency for shading
    label='Mean ± 1 Std Dev'
)

# Plot mean line
plt.plot(days, mean_values, color='blue', label='GraphCast Mean')

# Title and labels
plt.title('Mean Continuity Residual By Forecast Lead Time')
plt.xlabel('Forecast Lead Time [days]')
plt.ylabel('Continuity Residual [kg/m²/s]')
plt.legend()

# Add scientific notation for y-axis if values are very small
plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Save image with higher DPI for better quality
plt.savefig('mean_continuity_error_with_envelope_over_jan_w1.png', dpi=300, bbox_inches='tight')
plt.show()