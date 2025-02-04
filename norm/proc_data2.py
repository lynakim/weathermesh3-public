import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import random
import os
#import deep.meshes as meshes
import meshes
from functools import partial
import torch
#from model import *

import traceback
from data import *
from datasets import *

from torch.utils.data.dataloader import default_collate


def get_latlon_input_core(date_hour, dt_h=3, is_test=True):
    mesh = meshes.LatLonGrid() # default values of subsamp=1 and source=era5-28 are good
    base = get_proc_path_base(mesh, extra='base', is_test=is_test, create_dir=True) + "/%04d%02d"% (date_hour.year, date_hour.month)
    os.makedirs(base, exist_ok=True)
    
    with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
        norm = pickle.load(f)
    
    for i in range(0,24,dt_h):
        print(f"processing {date_hour} {i}")
        date_hour = date_hour.replace(hour=i)
        cp = base + "/%d.npz" % (to_unix(date_hour))
        if os.path.exists(cp):
            print(f"already done {cp} skipping")
            continue
        sfc_data = []
        for var in core_sfc_vars:
            print(var)
            mean, std2 = norm[var]
            f = get_dataset(get_raw_nc_filename(var, date_hour), date_hours=[date_hour])
            f = downsamp(f, mesh.subsamp).transpose(1, 2, 0)[:720]
            f = (f - mean[np.newaxis, np.newaxis, :])/np.sqrt(std2)[np.newaxis, np.newaxis, :]
            sfc_data.append(f[:,:,0].astype(np.float16))
        sfc_data = np.array(sfc_data)
        sfc_data = sfc_data.transpose(1, 2, 0)

        data = []
        for var in core_pressure_vars:
            print(var)
            mean, std2 = norm[var]
            f = get_dataset(get_raw_nc_filename(var, date_hour), date_hours=[date_hour])[0, mesh.wh_lev,:,:]
            f = downsamp(f, mesh.subsamp).transpose(1, 2, 0)[:720]
            f = (f - mean[np.newaxis, np.newaxis, mesh.wh_lev])/np.sqrt(std2)[np.newaxis, np.newaxis, mesh.wh_lev]
            data.append(f.astype(np.float16))
        
        cpt = cp.replace(".npz", ".tmp.npz")
        data = np.array(data).transpose(1, 2, 0, 3)
        np.savez(cpt, pr=data, sfc=sfc_data)
        os.rename(cpt, cp)
        print(f"saved {cp}")
    return

def get_latlon_input_extra(date_hours, var, is_test=True):
    assert type(var) == str and var in noncore_sfc_vars, "this is for a single extra var at a time only, use get_latlon_input_core for all core vars"
    agg_hours = get_agg_hours(var)
    mesh = meshes.LatLonGrid() # default values of subsamp=1 and source=era5-28 are good
    cps = []
    for date_hour in date_hours:
        # same .nc file can span multiple months
        base = get_proc_path_base(mesh, extra=var, is_test=is_test, create_dir=True) + "/%04d%02d"% (date_hour.year, date_hour.month)
        os.makedirs(base, exist_ok=True)
        cp = base + "/%d.npz" % (to_unix(date_hour))
        cps.append(cp)
        
    if all([os.path.exists(cp) for cp in cps]):
       print(f"already done {cps[0]} to {cps[-1]} skipping")
       return
    
    #with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
    with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
        norm = pickle.load(f)

    # nc_filenames = [get_raw_nc_filename(var, date_hour) for date_hour in date_hours]
    # assert len(set(nc_filenames)) == 1, "all sfc vars should be from the same file"
    nc_filename = get_raw_nc_filename(var, date_hours[0])
    assert nc_filename == get_raw_nc_filename(var, date_hours[-1]), "all sfc vars should be from the same file"
    last_nc_filename = get_raw_nc_filename(var, date_hours[0]-timedelta(days=1)) if agg_hours > 1 else None
    f = get_dataset(nc_filename, date_hours=date_hours, agg_hours=agg_hours, last_nc_filename=last_nc_filename, is_logtp=var == 'logtp')
    f = downsamp(f, mesh.subsamp)[:,:720,:]
    mean, std2 = norm[var]
    f = (f - mean)/np.sqrt(std2)
    for i, cp in enumerate(cps):
        if len(f[i]) == 0:
            print(f"skipping {cp} because it has no data")
            continue
        sfc_data = np.array(f[i,:,:].astype(np.float16))    
        cpt = cp.replace(".npz", ".tmp.npz")
        np.savez(cpt, x=sfc_data)
        os.rename(cpt, cp)
        print(f"saved {cp}")
    return

# NEED TO SET BIG TRAIN DATES BASED ON VARIABLE - check below examples
vars = core_pressure_vars + core_sfc_vars
dt_h = 1
big_train_dates = get_dates((datetime(2024, 9, 1), datetime(2024, 10, 31)))

#vars = ["168_2d"] #, "45_tcc",] 
# dt_h = 1    
# ms = get_monthly(get_dates([(datetime(2024, 9, 1), datetime(2024, 10, 31))]))
# big_train_dates = [get_dates((ms[i], ms[i+1], timedelta(hours=1)))[:-1] for i in range(len(ms)-1)]

# for different periods with different dt_h
# ms2 = get_monthly(get_dates([(datetime(2008, 1, 1), datetime(2024, 6, 1))]))
# big_train_dates2 = [get_dates((ms2[i], ms2[i+1], timedelta(hours=1)))[:-1] for i in range(len(ms2)-1)]
# big_train_dates = big_train_dates + big_train_dates2

# vars = ['179_ttr'] #'15_msnswrf', 'logtp']
# vars = ["201_mx2t", "202_mn2t"]
# dt_h = 1
# big_train_dates = get_semi_monthly_windows(datetime(1976, 12, 1), datetime(1977, 5, 1), dt_h=dt_h)

def doit_core(d):
  try:
    #for hr in range(0,24,3):
        #get_latlon_input(d+timedelta(hours=hr), config=cfg, mesh=mesh, ret=False, use_cache=False) # set use_cache to False if reprocessing
    get_latlon_input_core(d, dt_h=dt_h, is_test=False) 
    return 1
  except:
    #print("failed", d)
    f = traceback.format_exc()
    if "HDF" in f:
        print(f.split("HDF error: '")[1].split("'")[0])
        return f.split("HDF error: '")[1].split("'")[0]
    else:
        print("uhh", f)
    return 0

def doit_extra(var, d):
    try:
        get_latlon_input_extra(d, var, is_test=False) 
        return 1
    except:
        f = traceback.format_exc()
        if "HDF" in f:
            return f.split("HDF error: '")[1].split("'")[0]
        else:
            print("uhh", f)
        return 0

pool = multiprocessing.Pool(16)
print(f'Processing {vars} from {big_train_dates[0] if type(big_train_dates[0])==datetime else big_train_dates[0][0]} '
      f'to {big_train_dates[-1] if type(big_train_dates[-1])==datetime else big_train_dates[-1][-1]}')

if vars == core_pressure_vars + core_sfc_vars:
    #doit_core(big_train_dates[0]); exit()
    s = list(tqdm(pool.imap_unordered(doit_core, big_train_dates), total=len(big_train_dates)))
    print("sum", sum([x for x in s if type(x) == int]))
    bad = [x for x in s if type(x) == str]
    print("bad", bad)
else:
    for var in vars:
        print(f'Processing {var} from {big_train_dates[0] if type(big_train_dates[0])==datetime else big_train_dates[0][0]} '
              f'to {big_train_dates[-1] if type(big_train_dates[-1])==datetime else big_train_dates[-1][-1]}')
        
        s = list(tqdm(pool.imap_unordered(partial(doit_extra, var), big_train_dates), total=len(big_train_dates)))
        #doit_extra(var, big_train_dates[0]); exit()
        print("sum", sum([x for x in s if type(x) == int]))
        bad = [x for x in s if type(x) == str]
        print("bad", bad)

