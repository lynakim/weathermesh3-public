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

Gzarr = xr.open_zarr('gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr')
gc_batch_subset = Gzarr.sel(time=slice('2020-01-01T00:00:00', '2020-01-01T12:00:00'))

gc_batch_subset = analyze_mass_conservation(gc_batch_subset)

# weight error by volume
gc_error_weighted = gc_batch_subset['continuity_error'] * (gc_batch_subset['volume']/gc_batch_subset['volume'].sum(dim=['lat', 'lon', 'level'])) 
gc_error_weighted_mean = gc_error_weighted.sum(dim=['lat', 'lon', 'level']).mean(dim='time')
gc_error_weighted_std = gc_error_weighted.sum(dim=['lat', 'lon', 'level']).std(dim='time')

# %%
