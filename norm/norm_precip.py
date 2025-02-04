import xarray as xr
import numpy as np
import random
import pickle
from pprint import pprint

with open('/fast/consts/normalization.pickle', 'rb') as f:
    norm = pickle.load(f)
pprint(norm)

dataset = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2")

precip = dataset['total_precipitation']

idxs = random.sample(list(range(precip.shape[0])), 2000)

ohp = precip[idxs].to_numpy()
print(ohp.shape)

nrm = np.log(np.maximum(ohp + 1e-7,0))
print("hey", np.mean(nrm), np.std(nrm))

norm['logtp'] = (np.array([np.mean(nrm)]), np.array([np.var(nrm)]))

with open('/fast/consts/normalization.pickle', 'wb') as f:
    pickle.dump(norm, f)
