# %%
# this script exists because I forgot to normalize in process_ncarhres.py

from collections import defaultdict
import itertools
import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import sys
sys.path.append('/fast/wbhaoxing/deep')
from meteo.tools.process_dataset import download_s3_file
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

from utils import levels_joank, levels_medium, levels_hres, levels_tiny, levels_full, CONSTS_PATH, core_pressure_vars, core_sfc_vars, levels_aurora

# %%
hres9km = np.load("/fast/proc/hres9km/f000/2016010100.npz")
# %%
hresrt = np.load("/fast/proc/hres_rt/f000/2024032300.npz")
# %%
era5 = np.load("/fast/proc/era5/f000/201601/1451606400.npz")
# %%
plt.imshow(hresrt["sfc"][:,:,2])
# %%
plt.imshow(hres9km["sfc"][:,:,2])
# %%
plt.hist(hres9km["sfc"][:,:,0].flatten(), bins=200, histtype='step')
plt.hist(era5["sfc"][:,:,0].flatten(), bins=200, histtype='step')
# %%
plt.hist(hres9km["sfc"][:,:,1].flatten(), bins=200, histtype='step')
plt.hist(era5["sfc"][:,:,1].flatten(), bins=200, histtype='step')
# %%
plt.hist(hres9km["sfc"][:,:,2].flatten(), bins=200, histtype='step')
plt.hist(era5["sfc"][:,:,2].flatten(), bins=200, histtype='step')
# %%
plt.hist(hres9km["sfc"][:,:,3].flatten(), bins=200, histtype='step')
plt.hist(era5["sfc"][:,:,3].flatten(), bins=200, histtype='step')
# %%
