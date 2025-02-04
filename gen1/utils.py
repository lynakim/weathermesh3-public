import builtins
import copy
from datetime import datetime, timezone, timedelta
import importlib
import inspect
from itertools import product
import numpy as np
import os
import pickle
import time
import torch
from types import SimpleNamespace

CONSTS_PATH = '/fast/consts'
PROC_PATH = '/fast/proc'
RUNS_PATH = '/fast/ignored'


levels_gfs = [10, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 975, 1000]
levels_tiny = [50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925, 1000] # Open data IFS, also weatherbench HRES
levels_full = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
levels_small = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 950, 1000]
levels_medium = [10, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 875, 900, 925, 950, 975, 1000]
levels_joank = [10, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 975, 1000]
assert len(levels_joank) == 25
levels_ecm1 = [10, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
levels_ecm2 = [10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
levels_hres = [10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 925, 950, 1000]

core_pressure_vars = ["129_z", "130_t", "131_u", "132_v", "133_q"]
core_sfc_vars = ["165_10u", "166_10v", "167_2t", "151_msl"]

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
    "170_sst": "sea_surface_temperature",
    "45_tcc": "total_cloud_cover",
    }

num2levels = {}
for levels in [levels_tiny, levels_full, levels_medium, levels_ecm1, levels_ecm2, levels_small]:
    if len(levels) in num2levels: continue
    num2levels[len(levels)] = levels

D = lambda *x: datetime(*x, tzinfo=timezone.utc)
dimprint = lambda *a, **b: None

def sizeprint(s,name = ""):
    dimprint(f"{name} size:", s.numel()*s.element_size()/2**20, "MiB")

def unnorm_output(x,y,model,dt,y_is_deltas=None,skip_x=False,pad_x=False, skip_y=False):
    # this actually unnorms the output and the input together
    if not skip_y:
        if y.meta.delta_info is not None:
            if y_is_deltas is not None:
                assert y_is_deltas == True, f"y_is_deltas is {y_is_deltas} but y.meta.delta_info is {y.meta.delta_info}"
            assert y.meta.delta_info == dt, f'dt mismatch: {y.meta.delta_info} != {dt}'
            x,y = unnorm_output_partial(x,y,model,dt)
        else: 
            x = x[...,:y.shape[-1]]
        y = unnorm(y,model.mesh)
        if skip_x:
            return None, y
    
    x = unnorm(x,model.mesh,pad_x=pad_x)
    return x,y

def unnorm_output_partial(x,y,model,dt):
    B,N1,N2,O = y.shape
    x = x[...,:O]
    if y.meta.delta_info is not None:
        assert y.meta.delta_info == dt, "dt mismatch"
        dnorm = model.decoders[str(dt)].delta_norm_matrix
        dnorm = dnorm.to(x.device)
        if not isinstance(x,torch.Tensor):
            dnorm = dnorm.to('cpu').numpy()
        y = x + y * dnorm
    return x,y

def unnorm(x,mesh,pad_x=False):
    if pad_x and x.shape[-1] < mesh.state_norm_stds.shape[-1]:
        x = torch.nn.functional.pad(x, (0,mesh.state_norm_stds.shape[-1]-x.shape[-1]), mode='constant', value=0)

    x = x*torch.Tensor(mesh.state_norm_stds).to(x.device) + torch.Tensor(mesh.state_norm_means).to(x.device)
    assert x.dtype == torch.float32, "Bro you can't unnorm in fp16"
    return x

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

def sdate2nix(s):
    if int(s) < 1970000000:
        return int(s)
    else:
        return to_unix(datetime.strptime(s, "%Y%m%d%H"))

def to_filename(nix,dt,tags=[], always_plus=False):
    dts = ''
    if dt != 0 or always_plus:
        dts = f'+{dt}'
    tagstr = ''
    if len(tags) > 0:
        tagstr = '.'+'.'.join(tags)
    return f'{get_date_str(nix)}{dts}{tagstr}'

class NeoDatasetConfig():
    def __init__(self,conf_to_copy=None,**kwargs):
        global CONSTS_PATH,PROC_PATH,RUNS_PATH
        self.CLOUD = False
        self.WEATHERBENCH = "John is only keeping this around to not break joan's scrips casue John is a really nice guy and he beleives in accomodations for the disabled"
        self.source = 'era5-28'
        self.input_levels = None
        self.levels = None
        self.PROC_PATH = PROC_PATH ; self.CONSTS_PATH = CONSTS_PATH ; self.RUNS_PATH = RUNS_PATH
        self.subsamp = 1
        self.extra_sfc_vars = []
        self.extra_sfc_pad = 0
        self.extra_pressure_vars = []
        self.output_only_vars = []
        self.is_output_only = False
        self.ens_num = None
        self.intermediate_levels = []
        if conf_to_copy is not None:
            self.__dict__.update(copy.deepcopy(conf_to_copy.__dict__))

        for k in self.extra_sfc_vars:
            #assert k in self.output_only_vars, "haven't tested this yet"
            if k not in self.output_only_vars: print(f'WARNING: {k} is in extra_sfc_vars but not output_only_vars')
        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a NeoDatasetConfig attribute"
            setattr(self,k,v)

        self.update()

    def update(self):
        global levels_full
        assert '-' in self.source, f'source must be of the form "era5-28" or "hres-13", not {self.source}'
        #self.pressure_vars = ["129_z", "130_t", "131_u", "132_v", "135_w", "133_q", "075_crwc", "076_cswc", "248_cc", "246_clwc", "247_ciwc"]
        numlev = int(self.source.split('-')[1])
        if self.levels is None:
            self.levels = num2levels[numlev]
        if self.input_levels is None:
            self.input_levels = self.levels
        assert len(self.input_levels) == numlev, f'levels must be {numlev} long for {self.source}, not {len(self.input_levels)}'
        if self.source == 'hres-13':
            assert len(self.input_levels) == len(levels_tiny), f'levels must be {len(levels_tiny)} long for hres'
        self.pressure_vars = core_pressure_vars + self.extra_pressure_vars
        self.sfc_vars = core_sfc_vars + self.extra_sfc_vars + ['zeropad']*self.extra_sfc_pad
        self.varlist = self.pressure_vars + self.sfc_vars
        self.full_varlist = []
        for v in self.pressure_vars:
            self.full_varlist = self.full_varlist + [v+"_"+str(l) for l in self.levels]
        self.full_varlist = self.full_varlist + self.sfc_vars
        if self.CLOUD: self.PROC_PATH = None

        self.accum_vars = ["029_mlspr", "030_mcpr", "031_msr", "055_mtpr"]
        self.n_levels = len(self.levels)
        self.n_pr_vars = len(self.pressure_vars)
        self.wh_lev = [levels_full.index(x) for x in self.levels] #which_levels
        self.n_pr_vars = len(self.pressure_vars)
        self.n_pr = self.n_levels * self.n_pr_vars
        self.n_sfc = len(self.sfc_vars)
        self.n_sfc_vars = self.n_sfc
        self.n_vars = self.n_pr + self.n_sfc
        #print(self.n_sfc,self.sfc_vars)

        #check_path(self.CONSTS_PATH) ; check_path(self.PROC_PATH) ; check_path(self.RUNS_PATH)
    def get_zeros(self):
        return np.zeros((1, len(self.lats), len(self.lons), self.n_vars), dtype=np.float32)


def save_instance(x,path,mesh,downsample_levels=False, is_ensemble=False):
    if isinstance(x,torch.Tensor):
        x = x.detach().cpu().numpy()
    if downsample_levels:
        newconf = NeoDatasetConfig(conf_to_copy=mesh.config,levels=levels_ecm2)
        newmesh = type(mesh)(newconf)
        wh_levnew = [mesh.config.levels.index(x) for x in levels_ecm2]
        xshape_new = list(x.shape[:-1]) + [newmesh.n_vars]
        xnew = np.zeros(xshape_new,dtype=x.dtype)
        for i,j in enumerate(wh_levnew):
            xnew[...,i*mesh.n_pr_vars:(i+1)*mesh.n_pr_vars] = x[...,j*mesh.n_pr_vars:(j+1)*mesh.n_pr_vars]
        xnew[...,-mesh.n_sfc:] = x[...,-mesh.n_sfc:]
        x = xnew
        mesh = newmesh
    js,hash = mesh.to_json()
    os.makedirs(os.path.dirname(path),exist_ok=True)
    if not is_ensemble:
        metapath = os.path.dirname(path)+f'/meta.{hash}.json'
    else:
        metapath = os.path.join(os.path.dirname(path), '..', '..', f'meta.{hash}.json')
    if os.path.exists(metapath):
        with open(metapath,'r') as f:
            js2 = f.read()
        assert js == js2, "metadata mismatch"
    else:    
        with open(metapath,'w') as f:
            f.write(js)
    if isinstance(x,torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.shape[0] == 1:
        x = x[0]
    else:
        assert len(x.shape) == 3, "Can not be multi batch"
    filepath= path+f".{hash}.npy"
    print("Saving to", filepath)
    np.save(filepath,x)
    return filepath

def to_unix(date):
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    else: 
        assert date.tzinfo == timezone.utc, "Joan never intended for this to be used with non-utc dates"
    return int((date - datetime(1970,1,1,tzinfo=timezone.utc)).total_seconds())

def load_state_norm(wh_lev, config, with_means=False):
    norms = pickle.load(open(f'{CONSTS_PATH}/normalization.pickle', 'rb'))
    for k,v in norms.items():
        mean, var = v
        norms[k] = (mean, np.sqrt(var))
    
    state_norm_matrix = []
    state_norm_matrix2 = []
    for i, v in enumerate(config.pressure_vars):
        # TODO: this doesn't include full levels!!
        state_norm_matrix2.append(norms[v][0][wh_lev])
        state_norm_matrix.append(norms[v][1][wh_lev])
    for i, s in enumerate(config.sfc_vars):
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

def mapll(func, lls):
    return [[func(item) for item in sl] for sl in lls]

def flattenll(l):
    return [item for sublist in l for item in sublist]

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

class TensorWithMetadata(torch.Tensor):
    #def __new__(cls, data, *args, **kwargs):
    #    instance = torch.Tensor.__new__(cls, data, *args, **kwargs)
    #    instance.meta = None
    #    return instance

    def to(self, *args, **kwargs):
        y = super().to(*args, **kwargs)
        if hasattr(self,'meta'):
            y = set_metadata(y,self.meta.valid_at)
        return y


def set_metadata(y,valid_at,no_overwrite=True):
    if no_overwrite:
        assert not hasattr(y,'meta'), "huh, I wasn't expecting there already to be meta info for this tensor"
    y = TensorWithMetadata(y)
    y.meta = SimpleNamespace(delta_info = None, valid_at = valid_at)
    return y
00
def TIMEIT(thresh=None,):
    def inner(func):
        def wrapper(*args, **kwargs):
            t = time.time()
            ret = func(*args, **kwargs)
            elapsed = time.time() - t
            if thresh is None or elapsed > thresh:
                class_name = func.__qualname__.split('.')[0] if '.' in func.__qualname__ else ''
                if class_name:
                    print(f"{class_name}.{func.__name__} took {elapsed:.6f}s")
                else:
                    print(f"{func.__name__} took {elapsed:.6f}s")
            return ret
        return wrapper
    return inner

def get_proc_path_base(mesh,extra=None):
    base = os.path.join(PROC_PATH,mesh.source.split("-")[0],(f'extra/{extra}' if extra is not None else ''))
    fhstr = 'f000'
    if mesh.ens_num is not None:
        base = os.path.join(base,str(mesh.ens_num))
    if fhstr in os.listdir(base):
        base = os.path.join(base,fhstr)
    return base

def get_proc_path_time(date,mesh,extra=None):
    base = get_proc_path_base(mesh,extra=extra)
    paths = [f'{base}/',f'{base}/{date.strftime("%Y%m")}/']
    fnames = [f'{to_unix(date)}.npz',f'{date.strftime("%Y%m%d%H")}.npz']
    for path in [s+e for s,e in product(paths,fnames)]:
        if os.path.exists(path):
            return path
    assert False, f'Could not find {date} with base {base} and extra {extra}'

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

class SourceCodeLogger:
    def get_source(self):
        return inspect.getsource(self.__class__)

def print(*args, only_rank_0=False, **kwargs): 
    #return builtins.print(*args, **kwargs)
    try: 
        m = importlib.import_module("train")
        r = m.dist.get_rank()
        if only_rank_0 and r != 0: return
        #r = '?'
    except ValueError: r = ''
    except ImportError: r = ''
    except Exception as e: raise e
    #builtins.print(f"{sys._getframe(1).f_code.co_filename}:{sys._getframe(1).f_lineno} -> ")
    builtins.print(f"[{(lambda n: f'{n.hour:02}:{n.minute:02}:{n.second:02}.{int(n.microsecond / 1e4):02}')(datetime.now())}|{r:02}]", *args, **kwargs)

def print_mem(s,dev='cuda:0'):
    return # [sic]
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
        # TODO: this doesn't include full levels!!
        delta_norm_matrix.append(np.sqrt(norms[v][1]))#[self.mesh.wh_lev])
    for i, s in enumerate(config.sfc_vars):
        if s in norms:
            delta_norm_matrix.append(np.sqrt(norms[s][1]))
        else:
            assert s in config.output_only_vars or s == 'zeropad' or config.is_output_only == True, f'{s} not in norms'
            delta_norm_matrix.append([1.])
    delta_norm_matrix = np.concatenate(delta_norm_matrix).astype(np.float32)
    return norms,delta_norm_matrix

def copy_metadata(x1,x2):
    if not hasattr(x1,'meta'):
        return x2
    x2 = TensorWithMetadata(x2)
    x2.meta = x1.meta
    return x2

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

DEFAULT_CONFIG = NeoDatasetConfig()