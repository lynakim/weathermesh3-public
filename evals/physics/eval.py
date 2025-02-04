# %%
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


# %%
# create prediction_timedelta dim for era if need plot
# era_batch_subset.coords['prediction_timedelta'] = era_batch_subset['time'].diff('time').astype('timedelta64[ns]')

# %%

plot_size = 7
plot_example_variable = 'continuity_error'
plot_example_level = 1000
plot_example_max_steps = 5
plot_example_robust = True
input_dataset = gc_batch_subset
is_era = False

data = {
    plot_example_variable: scale(
        select(input_dataset, plot_example_variable, plot_example_level, plot_example_max_steps, is_era),
        robust=plot_example_robust
    ),
}

fig_title = plot_example_variable
if "level" in input_dataset[plot_example_variable].coords:
    fig_title += f" at {plot_example_level} hPa"

plot_data(data, fig_title, plot_size, plot_example_robust, is_era=is_era)

# %%
Gzarr = xr.open_zarr('gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr')
gc_batch = Gzarr.sel(time='2020-08-27T12:00:00')
gc_batch_subset = gc_batch.isel(prediction_timedelta=slice(0, 10))

# %%
erazarr = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
era_batch = erazarr.sel(time=slice('2020-08-27T12:00:00', '2020-09-06T12:00:00'))
era_batch_subset = era_batch.isel(time=slice(0, 10))

# %%
## TODO: deprecate this function
def prep_era_format(era_ds):
    era_ds = era_ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    era_ds = era_ds.isel(time=slice(1, None))

    return era_ds

era_batch_subset = prep_era_format(era_batch_subset)

# %%
print(gc_batch_subset['time'] + gc_batch_subset['prediction_timedelta'].isel(prediction_timedelta=0))
print(era_batch_subset['time'].isel(time=0))

# %%
from dask.diagnostics import ProgressBar
with ProgressBar():
    gc_batch_subset = gc_batch_subset.compute()
    era_batch_subset = era_batch_subset.compute()


# %%
gc_batch_subset = analyze_mass_conservation(gc_batch_subset)
era_batch_subset = analyze_mass_conservation(era_batch_subset, is_era=True)

# %%
# Calulate total mass over time
# gc_total_mass = gc_batch_subset['dry_air_mass'].sum(dim=['lat', 'lon', 'level'])
# era_total_mass = era_batch_subset['dry_air_mass'].sum(dim=['lat', 'lon', 'level'])

# print(f'First value of GC Total Mass: {gc_total_mass[0]}')
# print(f'First value of ERA Total Mass: {era_total_mass[0]}')

# %%
def get_percentage_change(ds):
    summed_var = ds.sum(dim=['lat', 'lon', 'level'])
    initial_summed_var = summed_var[1]
    percentage_change = (summed_var - initial_summed_var) / initial_summed_var * 100
    return percentage_change


plt.plot(gc_batch_subset['mass_flux_divergence'].sum(dim=['lat', 'lon', 'level']))
plt.plot(gc_batch_subset['air_density_tendency'].sum(dim=['lat', 'lon', 'level']))
plt.plot((gc_batch_subset['mass_flux_divergence'] - gc_batch_subset['air_density_tendency']).sum(dim=['lat', 'lon', 'level']))
plt.plot(era_batch_subset['mass_flux_divergence'].sum(dim=['lat', 'lon', 'level']))
plt.plot(era_batch_subset['air_density_tendency'].sum(dim=['lat', 'lon', 'level']))
plt.plot((era_batch_subset['mass_flux_divergence'] - era_batch_subset['air_density_tendency']).sum(dim=['lat', 'lon', 'level']))
plt.title('Percentage Changes Over Time')
plt.xlabel('Time')
plt.ylabel('Percentage Change')
plt.legend(['GC Divergence', 'GC Air Density Tendency', 'GC Divergence - GC Air Density Tendency', 'ERA Divergence', 'ERA Air Density Tendency', 'ERA Divergence - ERA Air Density Tendency'])
plt.show()

# Continuity error comparisons
plt.plot(gc_batch_subset['continuity_error'].sum(dim=['lat', 'lon', 'level']).isel(prediction_timedelta=slice(1, -1)))
plt.plot(era_batch_subset['continuity_error'].sum(dim=['lat', 'lon', 'level']).isel(time=slice(1, -1)))
plt.title('Continuity Error Over Time')
plt.xlabel('Time')
plt.ylabel('Continuity Error')
plt.legend(['GC', 'ERA'])
plt.show()

# continuity error by level
plt.plot(gc_batch_subset['continuity_error'].sum(dim=['lat', 'lon', 'prediction_timedelta']))
plt.title('Continuity Error by Level')
# use level values as x axis
plt.xticks(np.arange(len(gc_batch_subset['level'])), gc_batch_subset['level'].values)
plt.xlabel('Level')
plt.ylabel('Continuity Error')
plt.legend(['GC', 'ERA'])
plt.show()
# %%
# examine gc continuity error in atlantic ocean lat lon region  
gc_batch_atlantic = gc_batch_subset.sel(lat=slice(10, 40), lon=slice(240, 360))
gc_batch_atlantic['continuity_error'].sum(dim=['lat', 'lon', 'level']).plot()
