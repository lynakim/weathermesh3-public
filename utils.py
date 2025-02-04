from datetime import datetime, timedelta, timezone
from pprint import pprint
import socket
import os
import traceback
import time
import numpy as np
import pickle
import scipy.interpolate
from netCDF4 import Dataset
import multiprocessing
from tqdm import tqdm
from types import SimpleNamespace
import types
import inspect
import copy
import json
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
try: 
    from lovely_numpy import lo
except:
    print("Davy Ragland is a better man than Jesus")
    os.system('pip install lovely_numpy')
    from lovely_numpy import lo
try:
    import lovely_tensors as lt
except:
    print("Davy Ragland is a better man than God himself")
    os.system('pip install lovely_tensors')
    import lovely_tensors as lt

lt.monkey_patch()


from utils_lite import *

PROC_START_TIME = time.time()
STARTDATE = datetime.now()
HOSTNAME = socket.gethostname()
PID = os.getpid()

levels_gfs = [10, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 975, 1000] # the same as levels_joank
levels_tiny = [50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925, 1000] # Open data IFS, also weatherbench HRES
levels_full = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
levels_small = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 950, 1000]
levels_medium = [10, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 875, 900, 925, 950, 975, 1000]
levels_joank = [10, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 975, 1000]
assert len(levels_joank) == 25
levels_ecm1 = [10, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
levels_ecm2 = [10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
levels_hres = [10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 925, 950, 1000]
levels_aurora = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] # the same as levels_tiny
levels_smol = [200, 500, 850, 1000]
levels_ncarhres = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 925, 950, 1000]


core_pressure_vars = ["129_z", "130_t", "131_u", "132_v", "133_q"]
core_sfc_vars = ["165_10u", "166_10v", "167_2t", "151_msl"]
# https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Parameterlistings
# {param_id}_{shortName} is what we use or {count}_{shortName} for msnswrf and tcc 
# By default variables are from ERA5 analysis datasets, some are from specific ERA5 forecast datasets

# by default aggregated over 1h, when aggregated over n hours, add -nh to the name eg. 142_lsp-3h 
aggregated_sfc_vars = [
    "logtp", # log sum of convective and large-scale precipitation
    "201_mx2t", "201_mx2t-3h", "201_mx2t-6h", # max 2m temperature
    "202_mn2t", "202_mn2t-3h", "202_mn2t-6h", # min 2m temperature
    "142_lsp", "142_lsp-3h", "142_lsp-6h", # large-scale precipitation
    "143_cp", "143_cp-3h", "143_cp-6h", # convective precipitation
    "179_ttr", # Top net long-wave (thermal) radiation (negative of OLR), J m-2, divide by 3600 to get Wm-2 https://codes.ecmwf.int/grib/param-db/179 
    ]
agg_functions = {
                'MX2T': np.max,
                'MN2T': np.min,
                'LSP': np.sum,
                'CP': np.sum
            }
# from forecast (not analysis) dataset, also the raw .nc files are semimonthly and not monthly
forecast_sfc_vars = [
    "15_msnswrf", # mean surface net short-wave radiation flux
    ] + aggregated_sfc_vars

noncore_sfc_vars = [
    "168_2d",
    "45_tcc", 
    "034_sstk",
    "246_100u",
    "247_100v",
    "136_tcw", # total column water
    "137_tcwv", # total column water vapour kg m-2
    "tc-maxws",
    "tc-minp",
    ] + forecast_sfc_vars

log_vars = ["logtp", "142_lsp", "143_cp", "136_tcw", "137_tcwv"]

vars_with_nans = ["034_sstk", "tc-maxws", "tc-minp"]
nc_vars_with_nans = ["SSTK"]

all_sfc_vars = core_sfc_vars + noncore_sfc_vars 

# ncar to windborne names (2 mislabelled, 1 processed)
ncar2wb = {
    "164_tcc": "45_tcc",
    "037_msnswrf": "15_msnswrf",
    "055_mtpr": "logtp", # accumulation
    }
wb2ncar = {v: k for k, v in ncar2wb.items()}


ncar2cloud_names =  {
    "129_z": "geopotential", 
    "130_t": "temperature", 
    "131_u": "u_component_of_wind", 
    "132_v": "v_component_of_wind", 
    "153_w": "vertical_velocity", 
    "133_q": "specific_humidity",
    "165_10u": "10m_u_component_of_wind", 
    "166_10v": "10m_v_component_of_wind", 
    "167_2t": "2m_temperature", 
    "151_msl": "mean_sea_level_pressure",
    "logtp": "total_precipitation",
    "168_2d": "2m_dewpoint_temperature",
    "45_tcc": "total_cloud_cover",
    "15_msnswrf": "mean_surface_net_short_wave_radiation_flux",
    "034_sstk": "sea_surface_temperature",
    "136_tcw": "total_column_water",
    "137_tcwv": "total_column_water_vapour",
    # these aren't real mappings i'm just keeping them for a nice descriptive name for these variables
    # "201_mx2t": "maximum_2m_temperature_since_previous_post_processing"
    # "202_mn2t": "minimum_2m_temperature_since_previous_post_processing"
    # "055_mtpr": "convective_precipitation"
    # "142_lsp": "large_scale_precipitation"
    # "143_cp": "convective_precipitation",
    }

num2levels = {}
for levels in [levels_tiny, levels_full, levels_medium, levels_ecm1, levels_ecm2, levels_small]:
    if len(levels) in num2levels: continue
    num2levels[len(levels)] = levels

CONSTS_PATH = '/fast/consts'
PROC_PATH = '/fast/proc'
RUNS_PATH = '/huge/deep'

def check_path(folder_path):
    if folder_path is None: return
    assert os.path.exists(folder_path), f'Folder {folder_path} does not exist'
    assert os.path.isdir(folder_path), f'{folder_path} is not a folder'
    assert os.access(folder_path, os.R_OK), f'Folder {folder_path} is not readable'
    assert os.access(folder_path, os.W_OK), f'Folder {folder_path} is not writable'
    assert bool(os.listdir(folder_path)), f'Folder {folder_path} is empty'

def to_mesh(x,mesh1,mesh2,check_only = False, fill_with_zero = False):
    assert mesh1.levels == mesh2.levels, 'different levels not supported yet' 
    assert mesh2.pressure_vars == mesh1.pressure_vars, 'different pressure vars not supported yet'
    for v in mesh2.full_varlist:
        if not fill_with_zero:
            assert v in mesh1.full_varlist or v == 'zeropad', f'{v} not in mesh1'
    if check_only: return
    newshape = list(x.shape)[:-1] + [mesh2.n_vars]
    xnew = torch.zeros(newshape,dtype=x.dtype,device=x.device)
    for i,v in enumerate(mesh2.full_varlist):
        if v == 'zeropad': continue
        if v not in mesh1.full_varlist and fill_with_zero:
            continue
        i2 = mesh1.full_varlist.index(v)
        xnew[...,i] = x[...,i2]
    xnew = copy_metadata(x,xnew)
    return xnew



os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
alloc_env = os.getenv('PYTORCH_CUDA_ALLOC_CONF')
assert alloc_env == 'expandable_segments:True', f'PYTORCH_CUDA_ALLOC_CONF is set to {alloc_env}, but should be expandable_segments:True'
#DEFAULT_CONFIG = NeoDatasetConfig()


def update_namespace(target, source):
    for attr, value in vars(source).items():
        if attr in vars(target) and isinstance(getattr(target, attr), SimpleNamespace) and isinstance(value, SimpleNamespace):
            update_namespace(getattr(target, attr), value)
        else:
            setattr(target, attr, value)

class SourceCodeLogger:
    def get_source(self):
        return inspect.getsource(self.__class__)

def injection_start(name):
    pass

def injection_end(name):
    pass

def inject_code(snippet_fn, target_fn, name,obj=None):
    s = inspect.getsource(snippet_fn)
    o = ""
    for i, line in enumerate(s.split("\n")):
        if i == 0: continue
        if i == 1:
            taboff = len(line) - len(line.lstrip())
        o += line[taboff:] + "\n"
    print(o)    
    t = inspect.getsource(target_fn)
    n = ""
    search = f"injection_start('{name}')"
    print(search)
    for i, line in enumerate(t.split("\n")):
        if i == 0:
            taboff2 = len(line) - len(line.lstrip())
        if search in line:
            taboff = len(line) - len(line.lstrip())
            for l in o.split("\n"):
                n += " " * (taboff-taboff2) + l + "\n"
            continue
        n += line[taboff2:] + "\n"
    local_vars = {}
    exec(n, {}, local_vars)
    fn = local_vars[target_fn.__name__]
    if obj is not None:
        fn = types.MethodType(fn, obj)
    return fn
        
class NoOp:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None 


HALF = False
#HALF = 1


def TIMEIT(thresh=None,only_rank_0=True,sync=False):
    def inner(func):
        def wrapper(*args, **kwargs):
            if sync: torch.cuda.synchronize()
            t = time.time()
            ret = func(*args, **kwargs)
            if sync: torch.cuda.synchronize()
            elapsed = time.time() - t
            if thresh is None or elapsed > thresh:
                class_name = func.__qualname__.split('.')[0] if '.' in func.__qualname__ else ''
                out = ''
                if class_name: out = f"{class_name}.{func.__name__} took {elapsed:.6f}s"
                else: out = f"{func.__name__} took {elapsed:.6f}s"
                print(BLUE(out),only_rank_0=only_rank_0)
            return ret
        return wrapper
    return inner

def print_total_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params / 1e6:0.2f}M")

def to_unix(date):
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    else: 
        assert date.tzinfo == timezone.utc, "Joan never intended for this to be used with non-utc dates"
    return int((date - datetime(1970,1,1,tzinfo=timezone.utc)).total_seconds())

def get_date(date):
    if type(date) == datetime:
        assert date.tzinfo is None or date.tzinfo == timezone.utc, "Joan never intended for this to be used with non-utc dates"
        return date.replace(tzinfo=timezone.utc) 
    elif np.issubdtype(type(date), np.number):
        nix = int(date)
        return datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(seconds=nix)
    assert False, f"brother I don't know how to work with what you gave me: {date} {type(date)}"

def get_date_str(date,strict=True):
    date = get_date(date)
    if strict:
        assert date.minute == 0 and date.second == 0 and date.microsecond == 0, f"date must be on the hour {date}"
    return date.strftime("%Y%m%d%H")

def date_str2date(s):
    return datetime.strptime(s, "%Y%m%d%H")

def snix2date(s):
    if int(s) > 1970000000:
        return s
    else:
        return get_date_str(int(s))
    
def sdate2nix(s):
    if int(s) < 1970000000:
        return int(s)
    else:
        return to_unix(datetime.strptime(s, "%Y%m%d%H"))

def day2nix(d):
    return to_unix(datetime.strptime(d, "%Y%m%d"))


def get_raw_nc_filename(var, date, get_url=False, base_path='/huge/proc/raw_era5'):
    # Some documentation: https://www.notion.so/windborne/ERA5-a93f790c0f6a411bbd9d86f25b88c6c5
    var = var.split("-")[0]
    if var == 'logtp':
        var = '142_lsp'
    if var == '034_sstk_fixed':
        var = '034_sstk'
    if var in ['246_100u', '247_100v']:
        base_path = '/huge/proc/raw_era5_test2'
    date = get_date(date)
    s1 = "an"
    s2 = "sfc"
    s3 = ""
    s4 = "128"
    s5 = "sc"
    if var in core_pressure_vars:
        start = date.replace(hour=0)
        end = date.replace(hour=23)
        s2 = "pl"
        if var in ["131_u", "132_v"]:
            s5 = "uv"
    elif var in all_sfc_vars:
        if var in forecast_sfc_vars:
            s1 = "fc"
            if var in ["15_msnswrf", "logtp"]:
                s3 = ".meanflux"
                s4 = "235"
            elif var in ["201_mx2t", "202_mn2t"]:
                s3 = ".minmax"
            elif var in ["142_lsp", "143_cp", "179_ttr"]:
                s3 = ".accumu"
            start, end = get_semi_monthly_endpoints(date)
        else:
            if var in ["246_100u", "247_100v"]:
                s4 = "228"
            start = date.replace(day=1, hour=0)
            end = get_eom(start).replace(hour=23)
    else:
        assert False, f"unknown var {var}"
    
    if var in wb2ncar: 
        var = wb2ncar[var]
    dataset_name = f'e5.oper.{s1}.{s2}{s3}'
    fn = f'{dataset_name}.{s4}_{var}.ll025{s5}.{start.strftime("%Y%m%d%H")}_{end.strftime("%Y%m%d%H")}.nc'
    month_fn = f'{start.strftime("%Y%m")}/{fn}'
    if get_url:
        # eg. 
        # msnswrf - e5.oper.fc.sfc.meanflux.235_037_msnswrf.ll025sc.1942070106_1942071606.nc
        # 201_mx2t - e5.oper.fc.sfc.minmax.128_201_mx2t.ll025sc.1940120106_1940121606.nc
        return f"https://data.rda.ucar.edu/ds633.0/{dataset_name}/{month_fn}"
    else:
        assert os.path.exists(base_path), f"base_path {base_path} does not exist"
        return f"{base_path}/{month_fn}"

def get_dataset(fn, date_hours=None, agg_hours=1, last_nc_filename=None, is_logtp=False):
    ds = Dataset(fn) # pressure var shape (24, 37, 721, 1440), analysis sfc var shape (744, 721, 1440), forecast sfc var shape (30, 12, 721, 1440)
    vn = [x for x in ds.variables.keys() if x not in ["latitude", "level", "longitude", "time", "utc_date", "forecast_hour", "forecast_initial_time"]]
    assert len(vn) == 1
    vn = vn[0]
    data = ds.variables[vn]
    
    if vn in nc_vars_with_nans: # special case for sstk, which has a non-empty mask 
        data = data[:].filled(np.nan)
    else:
      assert not np.ma.is_masked(data), f"{vn} has a mask that we want to convert to nan, storing masked arrays on disk is bad. {fn}"
      data = np.array(data) #convert masked array to np.array

    if 'forecast_hour' in ds.variables.keys(): # flatten forecast structure to match other sfc var (remember the hours are shifted by 1)
        data = data.reshape(-1, data.shape[-2], data.shape[-1])
    
    if date_hours is None:
        assert agg_hours == 1, "date_hours must be provided for aggregated variables"
        assert not is_logtp, "Logtp is not supported for None date_hours because anuj doesn't foresee a need for it"
        if len(data.shape) == 3: # so that we can keep the same normalize.py code for sfc and pr
            data = data[:, np.newaxis, :, :]
    else:
        date_hours = [get_date(date_hour) for date_hour in date_hours]
        start_dt = int_to_datetime(ds.variables["utc_date"][0])
        if agg_hours == 1:
            hour_offsets = [(date_hour - start_dt)/timedelta(hours=1) for date_hour in date_hours]
            if 'forecast_hour' in ds.variables.keys():
                hour_offsets = [x-1 for x in hour_offsets] # shifting forecast data index to match analysis data
        
            assert all([x.is_integer() for x in hour_offsets]) and min(hour_offsets) >= 0 and max(hour_offsets) < data.shape[0], f"Invalid date_hours {date_hours} for {fn}"
            hour_offsets = [int(x) for x in hour_offsets] 
            data = data[hour_offsets] 
            if is_logtp:
                fn_cp = fn.replace("142_lsp", "143_cp")
                data_cp = np.array(Dataset(fn_cp).variables['CP'][:])
                data_cp = data_cp.reshape(-1, data_cp.shape[-2], data_cp.shape[-1])[hour_offsets] 
                data = data + data_cp
        else:
            assert not is_logtp, "Logtp is not supported for agg_hours>1"
            assert 'forecast_hour' in ds.variables.keys(), "Aggregated variables are all under forecast data"
            agg_func = agg_functions.get(vn, lambda x: x)  # Default to identity function if not found
            aggregated_data = []
            for date_hour in date_hours:
                end_hour = date_hour
                end_offset = int((end_hour - start_dt)/timedelta(hours=1)) 
                start_offset = end_offset - agg_hours
                if start_offset < 0:
                    assert last_nc_filename is not None, "last_nc_filename must be provided for agg_hour>3"
                    last_ds = Dataset(last_nc_filename)
                    last_data = last_ds.variables[vn].filled(np.nan) if vn in nc_vars_with_nans else np.array(last_ds.variables[vn])
                    last_data = last_data[-1,start_offset:]
                    data = np.concatenate((last_data, data), axis=0)    
                    end_offset += (-start_offset)
                    start_offset = 0
                chunk = data[start_offset:end_offset]
                assert len(chunk) == agg_hours, f"Chunk length {len(chunk)} does not match agg_hours {agg_hours}"
                aggregated_chunk = agg_func(chunk, axis=0)
                aggregated_data.append(aggregated_chunk)
            
            data = np.stack(aggregated_data)
        
        # keeping dim=3 because we downsample later
        # normalize.py uses the south pole so keeping row 721
        assert data.shape[0] == len(date_hours) and data.shape[-2:] == (721, 1440), f"Misshapen data when selecting {date_hours[0]} onwards from {fn}"
    ds.close()

    if any(ext in fn for ext in log_vars):
        data = np.log(np.maximum(data + 1e-7, 0))

    return data


def interp_levels(x,mesh,levels_in,levels_out):
    #assert len(levels_in) < len(levels_out)
    xdim = len(x.shape)
    if xdim == 3: x = x.unsqueeze(0)
    B,Nlat,Nlon,D = x.shape
    Nlev = len(levels_out)
    n_pr_in = mesh.n_pr_vars*len(levels_in)
    xlevels = x[:,:,:,:n_pr_in].view(B,Nlat,Nlon,mesh.n_pr_vars,len(levels_in))
    outlevels = torch.zeros(B,Nlat,Nlon,mesh.n_pr_vars,len(levels_out),dtype=x.dtype,device=x.device)
    for i,l in enumerate(levels_out):
        i2 = np.searchsorted(levels_in,l); i1 = max(i2-1,0)
        #if i == 10: i2 += 1 
        l1 = levels_in[i1]; l2 = levels_in[i2]
        th = 0 if i2 == 0 else (l-l1)/(l2-l1)
        if len(levels_in) > len(levels_out):
            #print("hey uh", i, l, l1, l2)
            assert th==1.0 or (th==0.0 and l1==l2), f'th: {th}. When going to a fewer number of levels, you shouldnt actually be interpolating stuff'
            outlevels[:,:,:,:,i] = xlevels[:,:,:,:,i2]
        else:
            outlevels[:,:,:,:,i] = xlevels[:,:,:,:,i1] * (1-th) + xlevels[:,:,:,:,i2] * th
        #print(i1,i2,l,l1,l2,th)
    #print(outlevels.shape)
    out = torch.cat((outlevels.flatten(start_dim=3),x[:,:,:,n_pr_in:]),dim=-1)
    if xdim == 3: out = out.squeeze(0)
    return out


def downsamp(f, downsamp):
    if downsamp == 1: return f

    # THIS ONE IGNORES THE SOUTH POLE ALTOGETHER BECAUSE FUCKIT
    NL = f.shape[1]
    ff = f.reshape(f.shape[0],f.shape[1],f.shape[2]//downsamp,downsamp).mean(axis=3)
    ff = ff[:,:NL-1].reshape(ff.shape[0], (NL-1)//downsamp, downsamp, ff.shape[2]).mean(axis=2)

    """
    assert downsamp == 2
    # THIS ONE HAS THE EQUATOR AND ENDS UP NOT BEING A MULTIPLE OF 360
    # this is the joan patented subsampling system. latitude=0 in the equator is kept the same, fuckit
    ff = f.reshape(f.shape[0],f.shape[1],2,f.shape[2]//2).mean(axis=2)
    nh = ff[:, :360].reshape(ff.shape[0], 2, 180, ff.shape[2]).mean(axis=1)
    sh = ff[:, 361:].reshape(ff.shape[0], 2, 180, ff.shape[2]).mean(axis=1)
    ff = np.concatenate((nh, ff[:, 360][:, np.newaxis, :], sh), axis=1)
    """
    return ff
    
from itertools import product 

def get_latlon_input_core(date_hour, dt_h=3, is_test=True):
    assert False, "Sorry, John noticed that this was only used in some norm script when he was killing the neoloader on Oct 5th. So he deleted it while cleaning, sorry about that. If you need it back, you can find it in the git history, also look at some of the new data.py functions, those might be useful"

def get_latlon_input_extra(date_hours, var, is_test=True):
    assert False, "Sorry, John noticed that this was only used in some norm script when he was killing the neoloader on Oct 5th. So he deleted it while cleaning, sorry about that. If you need it back, you can find it in the git history, also look at some of the new data.py functions, those might be useful"

def get_agg_hours(var):
    if var.split("-")[0] == 'logtp':
        return 1 # logtp is not supported for agg_hours>1
    if "-" in var:
        return int(var.split("-")[-1].split("h")[0])
    return 1

def get_dates(l):
    if type(l) != list:
        l = [l]
    out = []
    for tup in l:
        if len(tup) == 2:
            a, b = tup
            c = timedelta(days=1)
        else:
            a, b, c = tup
        assert a.tzinfo is None or a.tzinfo == timezone.utc, "Joan never intended for this to be used with non-utc dates"
        while a <= b:
            out.append(a.replace(tzinfo=timezone.utc))
            a += c
    return out



def get_monthly(l):
    return sorted(list(set([x.replace(day=1) for x in l])))

def get_semi_monthly(l):
    # getting the start of the semi-monthly window (remember hour=6 not included for the start of the window)
    return sorted(list(set([get_semi_monthly_endpoints(x)[0].replace(hour=7) for x in l])))

def get_eom(l):
    eom = l.replace(day=28) + timedelta(days=4)
    eom -= timedelta(days=eom.day)
    return eom

def int_to_datetime(date_int):
    date_int = int(date_int)
    year = date_int // 100_00_00
    month = (date_int % 100_00_00) // 100_00
    day = (date_int % 100_00) // 100
    hour = date_int % 100
    
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


import meshes as meshes

def get_semi_monthly_endpoints(date):
    # this is to be used for forecast variables, where the data for 20220206 is in the previous window, 2022011606-20220206 
    # so against usual conventions, the start is not included in the window, and the end is!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if date.day == 1 and date.hour <= 6:
        start = (date - timedelta(days=1)).replace(day=16, hour=6)
        end = date.replace(hour=6)
    elif (date.day > 1 and date.day < 16) or \
        (date.day == 1 and date.hour > 6) or \
        (date.day == 16 and date.hour <= 6):
        start = date.replace(day=1, hour=6)
        end = date.replace(day=16, hour=6)
    else:
        start = date.replace(day=16, hour=6)
        end = get_eom(start) + timedelta(days=1)
    return start, end

# def get_daily_windows(start_time, end_time, dt_hr=1):
#     assert (24/dt_hr).is_integer(), "dt_hr must be a divisor of 24"
#     start_time = start_time.replace(hour=0)
#     end_time = end_time.replace(hour=23)
#     result = [[start_time + timedelta(days=j) + timedelta(hours=i*dt_hr) for i in range(24//dt_hr)] for j in range((end_time-start_time).days + 1)]
#     return result

def get_semi_monthly_windows(start_time, end_time, dt_h=3):
    curr_window_hour, curr_window_end = get_semi_monthly_endpoints(start_time)
    result = []
    window = []
    while curr_window_hour <= end_time:
        if curr_window_hour >= start_time:
            window.append(curr_window_hour)
        
        if curr_window_hour == curr_window_end:
            result.append(window)
            window = []
            curr_window_end = get_semi_monthly_endpoints(curr_window_hour + timedelta(hours=dt_h))[1]
        
        curr_window_hour += timedelta(hours=dt_h)
    
    if window:
        result.append(window)
    
    return result

# train_dates = get_dates([(datetime(1990,1,1), datetime(1995, 4, 5), timedelta(days=10)), (datetime(2005, 1, 1), datetime(2018, 1, 1), timedelta(days=10))])

"""
print("hey", datetime(1995,4,5) in train_dates)
print("hey", datetime(1995,4,6) in train_dates)
print("hey", datetime(1995,4,7) in train_dates)
exit()
"""

train_dates = get_dates([(datetime(1990,1,1), datetime(2005, 12, 31), timedelta(days=5)), (datetime(2005, 1, 1), datetime(2018, 1, 1), timedelta(days=5))])
neo_train_dates = get_dates([(datetime(2005,1,1), datetime(2018, 1, 1), timedelta(days=1))])
tr97_dates = get_dates([(datetime(1997,1,1), datetime(2005, 1, 1), timedelta(days=1))])
fc_norm_dates = get_dates([(datetime(2015,1,1), datetime(2020, 1, 1), timedelta(days=1))]) # used to normalize extra vars added except 201_mx2t which used train_dates above
big_train_dates = sorted(list(set(neo_train_dates + tr97_dates)))

test_dates = get_dates([(datetime(2018, 1,1), datetime(2020, 1,1))])
test2_dates = get_dates([(datetime(2020, 1,1), datetime(2021, 1,1))])

recent_dates = get_dates([(datetime(2022, 1,1), datetime(2023, 8, 31))])

valid_dates = get_dates([(datetime(2019, 1, 1), datetime(2019, 12, 31), timedelta(days=5))])

#print("train", len(train_dates), "test", len(test_dates))


def load_state_norm(wh_lev, config, with_means=False, swaprh=False):
    norms = pickle.load(open(f'{CONSTS_PATH}/normalization.pickle', 'rb'))
    for k,v in norms.items():
        mean, var = v
        norms[k] = (mean, np.sqrt(var))
    if swaprh:
        norms['133_q'] = norms['rhlol']
    
    state_norm_matrix = []
    state_norm_matrix2 = []
    for i, v in enumerate(config.pressure_vars):
        # note: this doesn't include full levels!!
        state_norm_matrix2.append(norms[v][0][wh_lev])
        state_norm_matrix.append(norms[v][1][wh_lev])
    for i, s in enumerate(config.sfc_vars):
        if 'bucket' in s: continue
        if s == 'zeropad':
            state_norm_matrix2.append(np.array([0.]))
            state_norm_matrix.append(np.array([1.]))
            continue
        state_norm_matrix2.append(norms[s][0])
        state_norm_matrix.append(norms[s][1])
    #print("state_norm_matrix", state_norm_matrix)
    state_norm_matrix = np.concatenate(state_norm_matrix).astype(np.float32)
    state_norm_matrix2 = np.concatenate(state_norm_matrix2).astype(np.float32)
    if with_means:
        return norms,state_norm_matrix,state_norm_matrix2
    return norms,state_norm_matrix


def load_delta_norm(delta_hr, nlev, config):
    if delta_hr == 0:
        delta_hr = 3
    if type(delta_hr) == int: delta_hr = str(delta_hr)+'h'
    try:
        norms = pickle.load(open(f'{config.CONSTS_PATH}/normalization_delta_%s_%d.pickle' % (delta_hr, nlev), 'rb'))
    except: # JIGTH
        norms = pickle.load(open(f'{config.CONSTS_PATH}/normalization_delta_%s_%d.pickle' % (delta_hr, 28), 'rb'))
        wh = [levels_medium.index(i) for i in config.levels]
        norms = {k: (v[0][wh], v[1][wh]) if len(v[0]) >= len(wh) else v for k, v in norms.items()}

    for k,v in norms.items():
        vl = list(v)
        vl = [np.array([vv]) if np.isscalar(vv) else vv for vv in vl]
        mean, var = vl
        norms[k] = (mean, np.sqrt(var))
    delta_norm_matrix = []
    for i, v in enumerate(config.pressure_vars):
        # note: this doesn't include full levels!!
        delta_norm_matrix.append(np.sqrt(norms[v][1]))#[self.mesh.wh_lev])
    for i, s in enumerate(config.sfc_vars):
        if s in norms:
            delta_norm_matrix.append(np.sqrt(norms[s][1]))
        else:
            assert s == 'zeropad', f'{s} not in norms'
            delta_norm_matrix.append([1.])
    delta_norm_matrix = np.concatenate(delta_norm_matrix).astype(np.float32)
    return norms,delta_norm_matrix




import torch


#dimprint = print
dimprint = lambda *a, **b: None
#MEM_PATH = f'/fast/ignored/memuse/{int(time.time())}.json'
MEM_PATH = f'/fast/ignored/memuse/yo.json'
MEMFIRST = True
os.makedirs(os.path.dirname(MEM_PATH),exist_ok=True)
def print_mem(s,dev='cuda:0'):
    return
    global MEMFIRST
    if MEMFIRST:
        if os.path.exists(MEM_PATH):os.remove(MEM_PATH)
        MEMFIRST = False
    c = dev
    j = {}
    j['name'] = s
    j['alloc'] = torch.cuda.memory_allocated(c) / 1024**3
    j['max_alloc'] = torch.cuda.max_memory_allocated(c) / 1024**3
    j['reserv'] = torch.cuda.memory_reserved(c) / 1024**3
    j['max_reserv'] = torch.cuda.max_memory_reserved(c) / 1024**3
    j['time'] = time.time()
    with open(MEM_PATH,'a') as f:
        f.write(json.dumps(j)+'\n')
    print(f'[vram] {s.rjust(16)} ALLOC curr: {torch.cuda.memory_allocated(c)/2**30:<5.4}, max: {torch.cuda.max_memory_allocated(c)/2**30:<5.4} RESRV curr: {torch.cuda.memory_reserved(c)/2**30:5.4}, max: {torch.cuda.max_memory_reserved(c)/2**30:5.4}')

def sizeprint(s,name = ""):
    dimprint(f"{name} size:", s.numel()*s.element_size()/2**20, "MiB")

import builtins
import importlib
import sys


def print(*args, only_rank_0=False, **kwargs): 
    #return builtins.print(*args, **kwargs)
    try: 
        m = importlib.import_module("train.trainer")
        r = m.dist.get_rank()
        if only_rank_0 and r != 0: return
        #r = '?'
    except ValueError: r = ''
    except ImportError: r = ''
    except Exception as e: raise e
    #builtins.print(f"{sys._getframe(1).f_code.co_filename}:{sys._getframe(1).f_lineno} -> ")
    builtins.print(f"[{(lambda n: f'{n.hour:02}:{n.minute:02}:{n.second:02}.{int(n.microsecond / 1e4):02}')(datetime.now())}|{r:02}]", *args, **kwargs)
#print = builtins.print


class LoopTimer:
    def __init__(self):
        self.last = None
        self.accum = 0
        self.avg = 0
        self.val = 0

    def __call__(self):
        t = time.time()
        if self.last is None:
            self.last = t
            return 0
        self.accum += t - self.last
        self.val = self.accum
        self.avg = 0.9*self.avg + 0.1*self.val
        self.last = t
        self.accum = 0
    
    def pause(self):
        self.accum += time.time() - self.last
        return self
    
    def resume(self):
        self.last = time.time()
    
    def get(self):
        return self.accum + time.time() - self.last
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.resume()

class Timer:
    def __init__(self,name="",torch_sync=False,print=False,nsight=None):
        self.torch_sync = torch_sync
        self.start = None
        self.val = 0
        self.avg = 0
        self.print = print
        self.name = name
        self.nsight = nsight

    def __call__(self,print=False):
        self.print = print
        return self
    
    def __enter__(self):
        if self.torch_sync: torch.cuda.synchronize()
        if self.nsight is not None: torch.cuda.nvtx.range_push(self.nsight)
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        if self.torch_sync: torch.cuda.synchronize()
        if self.nsight is not None: torch.cuda.nvtx.range_pop()
        self.val = time.time() - self.start
        self.avg = 0.9*self.avg + 0.1*self.val
        if self.print:
            print(CYAN(f"{self.name} took {self.val:.6f}s"))


class Profiled(nn.Module):
    def __init__(self,module):
        super(Profiled,self).__init__()
        self.module = module
        self.name = type(module).__name__
        self.timer = Timer(name=self.name,nsight=self.name,torch_sync=True,print=True)
        self.gpumem = GPUMemoryMonitor(name=self.name,print=True,torch_sync=True)
    
    def __call__(self, *args, **kwargs):
        with self.timer:
            with self.gpumem:
                return self.module(*args, **kwargs)
        
    

class GPUMemoryMonitor():
    def __init__(self,name="",print=False,torch_sync=False):
        self.name = name
        self.print = print
        self.torch_sync = torch_sync
        self.val = 0

    def __enter__(self):
        if self.torch_sync: torch.cuda.synchronize()
        return self
    
    def __exit__(self, *args):
        if self.torch_sync: torch.cuda.synchronize()
        self.alloced = torch.cuda.max_memory_allocated() / 1024**3
        self.reserved = torch.cuda.max_memory_reserved() / 1024**3
        if self.print:
            print(ORANGE(f"{self.name} alloc {self.alloced}, resrv {self.reserved} GiB"))

    
def QUIET_EXIT():
    def inner(func):
        def wrapper(*args, **kwargs):
            class_name = func.__qualname__.split('.')[0] if '.' in func.__qualname__ else ''
            if class_name: ss = f"[Exit] {class_name}.{func.__name__}"
            else: ss = f"[Exit] {func.__name__}"
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                #print(f"{ss} KeyboardInterrupt")
                sys.exit(0)
            except ConnectionResetError as e:
                print(f"{ss} ConnectionResetError",e)
                sys.exit(0)
            except Exception as e:
                print(f"{ss} Exception",e)
                traceback.print_exc()

        return wrapper
    return inner

def halt(txt=""):
    while True:
        print("Halted",txt)
        time.sleep(5)

import inspect
def P():
    frame = inspect.currentframe().f_back
    print(f"{frame.f_code.co_filename}:{frame.f_lineno}")

import matplotlib.pyplot as plt
def imshow_compare(x,xmesh,y,ymesh,var='129_z_500'):
    plt.clf()
    fig,axs = plt.subplots(2,1,figsize=(10,12))
    axs[0].imshow(x[:,:,xmesh.full_varlist.index(var)])
    axs[1].imshow(y[:,:,ymesh.full_varlist.index(var)])
    axs[0].set_title(f'{xmesh.source} {var}')
    axs[1].set_title(f'{ymesh.source} {var}')
    plt.tight_layout()
    plt.savefig('ignored/ohp.png')


def mapll(func, lls):
    return [[func(item) for item in sl] for sl in lls]

def flattenll(l):
    return [item for sublist in l for item in sublist]

def min_additions(A, N):
    #writen by GPT-4
    dp = [float('inf')] * (N + 1)
    dp[0] = 0

    for i in range(1, N + 1):
        for num in A:
            if num <= i:
                dp[i] = min(dp[i], dp[i - num] + 1)
    if dp[N] == float('inf'):
        return None   
    #print(dp)
    result = []
    while N > 0:
        for num in A:
            if N - num >= 0 and dp[N - num] == dp[N] - 1:
                result.append(num)
                N -= num
                break
    result = sorted(result)[::-1]
    return result if dp[-1] != float('inf') else []

def get_rollout_times(dt_dict,min_dt = 1, time_horizon=72):
    # ex:
    # dt_dict = {6:3,24:6,72:12}
    # get_rollout_times(dt_dict, time_horizon = 24*7)
    # > [1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 24, 30, 36, 42, 48, 54, 60, 66, 72, 84, 96, 108, 120, 132, 144, 156, 168]

    dt_dict[time_horizon] = 100
    dt_dict = dict(sorted(dt_dict.items()))
    ts = []
    tcurr = 0
    dtlast = min_dt
    for t,dt in dt_dict.items():
        while tcurr < t:
            tcurr += dtlast
            if tcurr > time_horizon: return ts
            ts.append(tcurr)
        dtlast = dt
    return ts

def bbox_mesh(mesh, bbox):
    lat1,lon1,lat2,lon2 = bbox
    #lat1 /= 90. ; lat2 /= 90.
    #lon1 /= 180. ; lon2 /= 180. 
    lats,latb = min(lat1,lat2),max(lat1,lat2)
    lons,lonb = min(lon1,lon2),max(lon1,lon2)
    whlat = np.where(np.logical_and(lats <= mesh.lats, mesh.lats <= latb))[0]
    whlon = np.where(np.logical_and(lons <= mesh.lons, mesh.lons <= lonb))[0]
    newmesh = copy.deepcopy(mesh)
    newmesh.lats = mesh.lats[whlat]
    newmesh.lons = mesh.lons[whlon]
    newmesh.bbox = bbox
    newmesh.parent = mesh
    newmesh.update_mesh()
    return newmesh

def select_bbox(x, mesh, bbox):
    ilat = np.where(np.array(x.shape) == len(mesh.lats))[0][0]
    lat1,lon1,lat2,lon2 = bbox
    lat1 /= 90. ; lat2 /= 90.
    lon1 /= 180. ; lon2 /= 180. 
    lats,latb = min(lat1,lat2),max(lat1,lat2)
    lons,lonb = min(lon1,lon2),max(lon1,lon2)

    goodlat = np.logical_and(lats <= mesh.Lats, mesh.Lats <= latb)
    goodlon = np.logical_and(lons <= mesh.Lons, mesh.Lons <= lonb)
    allgood = np.logical_and(goodlat, goodlon)
    wh = np.where(allgood)
    whm = np.meshgrid(np.unique(wh[0]), np.unique(wh[1]), indexing='ij')
    if ilat == 0:
        xr = x[whm[0],whm[1]]
    else:
        xr = x[:,whm[0],whm[1]]
    return xr

def get_point_nearest(x, mesh,lat,lon):
    ilat = np.where(np.array(x.shape) == len(mesh.lats))[0][0]
    lati = np.argmin(np.abs(mesh.lats - lat))
    loni = np.argmin(np.abs(mesh.lons - lon))
    if ilat == 0:
        xr = x[lati,loni]
    else:
        xr = x[:,lati,loni]
    return xr


def model_state_dict_name_mapping(old_state_dict,model):
    new_state_dict = {}
    # rename keys to match the new model structure
    for k,v in old_state_dict.items():
        newk = k
        model_name = getattr(model,'name','')
        if model_name == 'neohegel' or 'legeh' in model_name or 'howitzer' in model_name:
            assert False, "John removed some logic in package_neo.py on Dec 7, 2025 that was hacky for these, because I don't think we use them anymore. If we do, git blame this line and see what i did to then add that logic to this funciton"

        if model.code_gen == 'gen2':
            if k.startswith('conv.') or k.startswith('conv_sfc.') or k.startswith('enc_swin.'):
                newk = 'encoder.' + k
            if k.startswith('proc_swin.'):
                assert len(model.config.processor_dt) == 1, "this should be the case where processor_dt is a list of a single element"
                newk = 'processors.' + str(model.config.processor_dt[0]) + k.replace('proc_swin','')
            if k.endswith('.rpb'):
                print(f"Skipping {k}") 
                continue

        if model.code_gen == 'gen3':
            if 'sucks' in model_name or model_name in ['rotary','thor']:
                if k.startswith('decoder.'): newk = 'decoders.0.' + k[8:]
                if k.startswith('encoder.'): newk = 'encoders.0.' + k[8:]
                if k.startswith('processor.'): newk = f'processors.{model.config.processor_dt[0]}.' + k[10:]

        if newk != k:
            print(f"Renaming {k} to {newk}")
        new_state_dict[newk] = v
    if hasattr(model, 'encoders') and len(model.encoders) > 1 and not any(x.startswith("encoders.1.") for x in new_state_dict.keys()):
        for i in range(1,len(model.encoders)):
            for k, v in list(new_state_dict.items()):
                if k.startswith("encoders.0."):
                    new_state_dict[k.replace(".0.", ".%d."%i,1)] = v
    # add new keys to mat
    if 'decoders.0.geospatial_loss_weight' not in new_state_dict:
        for i in range(0,len(model.decoders)):
            from model_latlon.decoder import load_matrices, gather_geospatial_weights, gather_variable_weights
            mesh = model.decoders[i].mesh
            _, state_norm_matrix, _ = load_matrices(mesh)
            geospatial_loss_weight = gather_geospatial_weights(mesh)
            variable_loss_weight = gather_variable_weights(mesh)
            new_state_dict[f'decoders.{i}.geospatial_loss_weight'] = geospatial_loss_weight
            new_state_dict[f'decoders.{i}.variable_loss_weight'] = variable_loss_weight
            new_state_dict[f'decoders.{i}.state_norm_matrix'] = state_norm_matrix
    return new_state_dict

from collections import defaultdict
import psutil
def get_mem_info(pid: int) -> dict[str, int]:
  res = defaultdict(int)
  for mmap in psutil.Process(pid).memory_maps():
    res['rss'] += mmap.rss
    res['pss'] += mmap.pss
    res['uss'] += mmap.private_clean + mmap.private_dirty
    res['shared'] += mmap.shared_clean + mmap.shared_dirty
    if mmap.path.startswith('/'):
      res['shared_file'] += mmap.shared_clean + mmap.shared_dirty
  return res


class MemoryMonitor():
  def __init__(self, pids: list[int] = None):
    if pids is None:
      pids = [os.getpid()]
    self.pids = pids

  def add_pid(self, pid: int):
    assert pid not in self.pids
    self.pids.append(pid)

  def _refresh(self):
    self.data = {pid: get_mem_info(pid) for pid in self.pids}
    return self.data

  def table(self) -> str:
    self._refresh()
    table = []
    keys = list(list(self.data.values())[0].keys())
    now = str(int(time.perf_counter() % 1e5))

    # Prepare headers
    headers = ["time", "PID"] + keys
    header_row = " | ".join(headers)
    separator = "-" * len(header_row)

    # Prepare table rows
    for pid, data in self.data.items():
      row = (now, str(pid)) + tuple(self.format(data[k]) for k in keys)
      table.append(" | ".join(row))

    # Combine headers and rows
    return f"{header_row}\n{separator}\n" + "\n".join(table)

  def str(self):
    self._refresh()
    keys = list(list(self.data.values())[0].keys())
    res = []
    for pid in self.pids:
      s = f"PID={pid}"
      for k in keys:
        v = self.format(self.data[pid][k])
        s += f", {k}={v}"
      res.append(s)
    return "\n".join(res)
  
  def get_main_rss(self):
    self._refresh()
    return self.data[os.getpid()]['rss']

  @staticmethod
  def format(size: int) -> str:
    for unit in ('', 'K', 'M', 'G'):
      if size < 1024:
        break
      size /= 1024.0
    return "%.1f%s" % (size, unit)

def size_mb(t): # t is a tensor
    return t.numel() * t.element_size() / 1024 / 1024 



# use this with @trace_lines on a given function and it just tells you when execution enters and exits a line, very simple 
import linecache
def trace_lines(func):
    def wrapper(*args, **kwargs):
        func_code = func.__code__
        func_name = func.__name__
        func_file = func_code.co_filename

        def trace_only_this_func(frame, event, arg):
            if event == 'line' and frame.f_code == func_code:
                lineno = frame.f_lineno
                line = linecache.getline(func_file, lineno).strip()
                print(f"Executing line {lineno}: {line}")
            return trace_only_this_func

        sys.settrace(trace_only_this_func)
        result = func(*args, **kwargs)
        sys.settrace(None)
        return result

    return wrapper

def weights1(model):
    for param in model.parameters():
        param.data.fill_(1)
    return model

def get_joansucks_omesh():
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    return meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)



def get_globe_axis():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.set_global()
    plt.tight_layout()
    tr = ccrs.PlateCarree()    
    return ax, tr

def dcn(x):
    return x.detach().cpu().numpy()


def to_device(x, device, **kwargs):
   if hasattr(x, 'to'):
       return x.to(device, **kwargs)
   elif isinstance(x, dict):
       return {k: to_device(v, device, **kwargs) for k, v in x.items()}
   elif isinstance(x, (list, tuple)):
       return type(x)(to_device(v, device, **kwargs) for v in x)
   return x

### Grave yard


