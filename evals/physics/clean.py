#%%
import xarray as xr
from eval_helpers import analyze_mass_conservation   

def load_zarr_to_numpy(path, time):
    """
    Load a Zarr dataset from the given path and return a dictionary
    of numpy arrays corresponding to the data variables.

    Parameters:
    - path (str): Path to the Zarr dataset.

    Returns:
    - dict: A dictionary where keys are variable names and values are numpy arrays.
    """
    # Open the Zarr dataset using xarray
    ds = xr.open_zarr(path)
    ds = ds.sel(time=time)
    
    # Convert each variable to a numpy array and store in a dictionary
    data_dict = {var_name: var.data.compute() if hasattr(var.data, 'compute') else var.data
                 for var_name, var in ds.data_vars.items()}
    
    coords_dict = {coord_name: coord.data.compute() if hasattr(coord.data, 'compute') else coord.data
                 for coord_name, coord in ds.coords.items()}
    
    return data_dict, coords_dict

# %%
cursed_batch = load_zarr_to_numpy('gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr', '2020-08-27T12:00:00')
# %%
class ShittyXarray:
    def __init__(self, data_dict, coords_dict):
        self.data = data_dict
        self.coords = coords_dict

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        elif key in self.coords:
            return self.coords[key]
        else:
            raise KeyError(f"Key {key} not found in data or coords")

# %%

ds = ShittyXarray(*cursed_batch)
ds = analyze_mass_conservation(ds)

# %%
R_d = 287.1 # Dry gas constant (J/(kg⋅K))
R_v = 461.5 # Water vapor gas constant (J/(kg⋅K))
x = ds['specific_humidity']
print(x.shape, type(x))
assert False
out = (1 - x + x * R_d/R_v)
# %%

import os
import numpy as np
from numpy.distutils.system_info import get_info

print("env vars for parallelization")
print(f"OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS')}")
print(f"MKL_NUM_THREADS: {os.getenv('MKL_NUM_THREADS')}")
print(f"OPENBLAS_NUM_THREADS: {os.getenv('OPENBLAS_NUM_THREADS')}")
print(f"BLIS_NUM_THREADS: {os.getenv('BLIS_NUM_THREADS')}")

print(f"Numpy config: {np.show_config()}")

print("BLAS info")
print(get_info("blas"))
# %%
print(get_info("blas"))


# %%
np.__version__
# %%
np.show_config()
# %%
