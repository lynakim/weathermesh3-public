from utils import *
import torch
import json
from meshes import BalloonData, LatLonGrid, TCRegionalIntensities, DAMesh, StationData, SurfaceData, SatwindData, RadiosondeData, WindborneData
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict
from functools import cache, reduce, partial
from pandas import date_range
import pathlib 
from pysolar.solar import get_altitude_fast
import diskcache as dc
cache = dc.Cache('/tmp/data_tots/')

# hard coded for now. Lower numbers will be favored

Gnorms = None
def get_norms(mesh):
    global Gnorms
    if Gnorms is None:
        Gnorms, _ = load_state_norm(mesh.wh_lev,mesh) # this is for weatherbench only
    return Gnorms

def get_proc_path_base(mesh,load_location=None,extra='base', is_test=False, create_dir=False):
    if load_location == 'weatherbench': 
        return load_location
    mesh_source_path = mesh.source.split("-")[0] + is_test*'_test'
    loadloc = mesh.load_locations[0] if load_location is None else load_location
    base = os.path.join(loadloc,mesh_source_path,(f'extra/{extra}' if extra != 'base' else ""))
    if mesh.ens_num is not None:
        base = os.path.join(base,str(mesh.ens_num))
    if not os.path.exists(base):
        if not create_dir:
            return None
        os.makedirs(base, exist_ok=True)
    fhstr = 'f000'
    if fhstr in os.listdir(base) or extra == 'base': #added second case for when we're creating a new era5 folder
        base = os.path.join(base,fhstr)
    return base

def get_proc_path_time(date,mesh,load_location=None,extra='base'):
    # this works with both era5 and era5_daily by checking both possible path formats
    base = get_proc_path_base(mesh,load_location=load_location,extra=extra)
    paths = [f'{base}/',f'{base}/{date.strftime("%Y%m")}/', f'{base}/{date.strftime("%Y")}/']
    fnames = [f'{to_unix(date)}.npz',f'{date.strftime("%Y%m%d%H")}.npz', f'{date.strftime("%Y%m%d")}.npz']
    for path in [s+e for s,e in product(paths,fnames)]:
        if os.path.exists(path):
            return path
    assert False, f'Could not find {date} with base {base} and extra {extra} at load location {load_location}'

def seek(path, ext='.npy',fn=lambda x: to_unix(date_str2date(x))):
    if not os.path.exists(path):
        return np.array([],dtype=np.int64)
    return np.array([fn(f.name[:-len(ext)]) for f in os.scandir(path) if f.name[:-len(ext)].isdigit() and f.name.endswith(ext)], dtype=np.int64)

class AnalysisDataset():
    def __init__(self, mesh, is_required=True, load_locations=["/fast/proc/"], gfs=False):
        self.mesh = mesh
        self.all_var_files = mesh.all_var_files
        self.ens_num = mesh.ens_num
        self.source = mesh.source
        self.is_required = is_required
        self.load_locations = load_locations
        self.clamp_input = 13 
        self.clamp_output = 13
        self.gfs = gfs

    def summary_str(self):
        return self.mesh.summary_str()

    #@TIMEIT()
    @cache.memoize(expire=60*60*24)
    @staticmethod
    def get_file_times(base,datemin,datemax, gfs=False):
        # note that this function will actually return extra times outside the requested range.
        # this is okay, because we want to be able to adjust it in the get_loadable_times function.
        if gfs:
            ext = '.npz'
            filenames = np.array([f.name[:-len(ext)] for f in os.scandir(base) if f.name[:-len(ext)].isdigit() and f.name.endswith(ext)], dtype=np.int64)
            tots = np.sort(filenames)
        else:
            yymm = date_range(datemin-timedelta(days=31),datemax+timedelta(days=31),freq='ME').strftime("%Y%m")
            tots = np.sort(np.concat([partial(seek,ext='.npz',fn=int)(f'{base}/{ym}') for ym in yymm]))
        return tots
    
    def get_loadable_times(self, datemin, datemax):
        alldates = []
        for var in self.all_var_files: 
            varx = var if var != 'zeropad' else '45_tcc' # i don't understand this but w/e
            base = get_proc_path_base(self.mesh,load_location=self.load_locations[0],extra=varx)
            if base is None:
                return np.array([])
            dates = self.__class__.get_file_times(base,datemin,datemax, self.gfs and 'f000' in base)
            alldates.append(dates)
        dates = reduce(np.intersect1d,alldates)
        dates = dates + self.mesh.hour_offset*3600
        dates = dates[np.where(np.logical_and(to_unix(datemin) <= dates, dates <= to_unix(datemax)))]
        return dates

    def neo_get_latlon_input(self, date, is_output):
        mesh = self.mesh
        assert isinstance(mesh, LatLonGrid)
        load_loc = self.load_locations[0]

        if load_loc == 'weatherbench':
            pr,sfc = self.get_weatherbench(date)
            return pr,sfc
        
        # robust to /fast/proc going down
        while True:
            if os.path.exists(load_loc):
                break
            print(f"Waiting for {load_loc} to come online") # doesn't get printed for some reason
            time.sleep(10)
        
        cp = get_proc_path_time(date,mesh,load_location=load_loc)
        assert os.path.exists(cp), cp
        assert mesh.subsamp == 1, "John on Jan7 wasn't sure if we ever use subsamp > 1"
        if mesh.parent is None: 
            cf = np.load(cp)
            if mesh.levels == []:
                pr,sfc = None, cf["sfc"]
            elif cf["pr"].shape[0] == 721:
                pr,sfc = cf["pr"][:720], cf["sfc"][:720]
            elif cf["pr"].shape[0] == 1801:
                pr,sfc = cf["pr"][:1800], cf["sfc"][:1800]
            else:
                pr,sfc = cf["pr"], cf["sfc"]
        else: 
            assert False, "This is not yet supported"
        assert not np.isnan(pr).any(), "NaNs in pr for " + str(date.strftime("%Y-%m-%d"))

        assert mesh.extra_pressure_vars == [], "This is not yet supported"
        if len(mesh.extra_sfc_vars) > 0 or mesh.extra_sfc_pad > 0:
            assert mesh.parent is None, "This is not yet supported"
            sfc_total = np.zeros((sfc.shape[0],sfc.shape[1],len(mesh.sfc_vars)), dtype=np.float16)
            sfc_total[:,:,:len(core_sfc_vars)] = sfc
            # Zeroing the extra_sfc_pad variables implicitly, as they're part of sfc_vars
            for i,e in enumerate(mesh.extra_sfc_vars):
                if e == "zeropad":
                    #sfc_total[:,:,len(core_sfc_vars)+i] = 0
                    continue
                if "_bucket" in e:
                    if "_bucket0" not in e: continue
                    # ok we do everything from bucket 0 at once
                    base = e.split("_bucket")[0]
                    if base == "142_lsp":
                        mean, std = (np.array([-13.32673432]), np.array([8.47666774**0.5])) # lsp
                    elif base == "143_cp":
                        mean, std = (np.array([-13.55428366]), np.array([9.14669573**0.5])) # cp
                    else:
                        assert False, "??"
                    buckets_scaled = (np.log(mesh.precip_buckets+1e-7)-mean)/std
                    assert base in mesh.extra_sfc_vars
                    digits = np.digitize(sfc_total[:,:,len(core_sfc_vars)+mesh.extra_sfc_vars.index(base)],buckets_scaled)
                    B = len(buckets_scaled)
                    assert digits.max() < B
                    sfc_total[:,:, len(core_sfc_vars)+i:len(core_sfc_vars)+i+B] = np.eye(B)[digits]
                    continue

                load_loc = self.load_locations[0]
                cp = get_proc_path_time(date,mesh,load_location=load_loc,extra=e)
                assert os.path.exists(cp), cp
                cfe = np.load(cp)['x']
                assert (cfe.shape == sfc.shape[:-1]) or (cfe.shape[0] == 1 and cfe.shape[1:] == sfc.shape[:-1]), f"{cp} is shape {cfe.shape}, expected {sfc.shape[:-1]}"
                # convert nans to 0
                if not is_output: cfe = np.nan_to_num(cfe, nan=0.0)
                sfc_total[:,:,len(core_sfc_vars)+i] = cfe # *0 + 0.666
            
            sfc = sfc_total
            #print("sfc shape", sfc.shape)
        assert sfc.shape[-1] == len(mesh.sfc_vars), f"sfc shape {sfc.shape} doesn't match sfc_vars {mesh.sfc_vars}"
        return pr,sfc

    def load_data(self, nix, is_output=False):
        nix = nix - self.mesh.hour_offset*3600
        pr,sfc = self.neo_get_latlon_input(get_date(nix), is_output)
        mesh = self.mesh
        F = torch.HalfTensor
        xsfc = F(sfc)
        assert pr is not None or mesh.levels == [], f"Missing pr data for {mesh.source} at {nix}"
        if pr is not None:
            xpr = F(pr)
            x_pr_flat = torch.flatten(xpr, start_dim=-2)
            x = torch.cat((x_pr_flat, xsfc), -1)
        else:
            x = xsfc
        if len(mesh.input_levels) != len(mesh.levels) and x.shape[-1] != mesh.n_vars: #if x is already the right shape, don't do this
            last_lev = mesh.input_levels
            for inter_lev in mesh.intermediate_levels+[mesh.levels]:
                x = interp_levels(x,mesh,last_lev,inter_lev)
                last_lev = inter_lev
        clamp = self.clamp_output if is_output else self.clamp_input
        x = torch.clamp(x, -clamp, clamp)
        assert x.shape[-1] == mesh.n_vars, f"{x.shape} vs {mesh.n_vars}, {mesh.source}"
        return x 

class DailyAnalysisDataset(AnalysisDataset):
    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
    
    @cache.memoize(expire=60*60*24)
    @staticmethod
    def get_file_times(base,datemin,datemax):
        # note that this function will actually return extra times outside the requested range.
        # this is okay, because we want to be able to adjust it in the get_loadable_times function.
        yy = date_range(datemin-timedelta(days=31),datemax+timedelta(days=31),freq='ME').strftime("%Y")
        tots = np.sort(np.concat([partial(seek,ext='.npz',fn=day2nix)(f'{base}/{y}') for y in yy]))
        return tots

SECONDS_PER_YEAR = 31_556_925
def transform_data(x,t0,in_varlist,out_varlist,mesh):
    B,D = x.shape 
    assert D == len(in_varlist)
    F  = len(out_varlist)
    out = np.zeros([B,F],dtype=np.float32) + np.nan

    def get_abs_time():
        if 'reltime_seconds' in in_varlist:
            return t0 + x[:,in_varlist.index('reltime_seconds')].astype(np.int64) 
        elif 'reltime_hours' in in_varlist:
            return t0 + (x[:,in_varlist.index('reltime_hours')]* 3600).astype(np.int64)
        else:
            assert False, "No reltime variable found"

    abs_time = get_abs_time()

    for i,var in enumerate(out_varlist):
        #if var == 'sin(lat)': out[:,i] = np.sin(x[:,in_varlist.index('lat_deg')] * np.pi / 180)
        #elif var == 'cos(lat)': out[:,i] = np.cos(x[:,in_varlist.index('lat_deg')] * np.pi / 180)
        #elif var == 'sin(lon)': out[:,i] = np.sin(x[:,in_varlist.index('lon_deg')] * np.pi / 180)
        #elif var == 'cos(lon)': out[:,i] = np.cos(x[:,in_varlist.index('lon_deg')] * np.pi / 180)
        if var == 'sin(time_of_year)': out[:,i] = np.sin(t0 / SECONDS_PER_YEAR * 2 * np.pi)
        elif var == 'cos(time_of_year)': out[:,i] = np.cos(t0 / SECONDS_PER_YEAR * 2 * np.pi)
        elif var == 'sin(time_of_day)': out[:,i] = np.sin(abs_time / 3600 / 24 * 2 * np.pi)
        elif var == 'cos(time_of_day)': out[:,i] = np.cos(abs_time / 3600 / 24 * 2 * np.pi)
        elif var == 'sin(satzenith_deg)': out[:,i] = np.sin(x[:,in_varlist.index('satzenith_deg')] * np.pi / 180)
        elif var == 'cos(satzenith_deg)': out[:,i] = np.cos(x[:,in_varlist.index('satzenith_deg')] * np.pi / 180)
        elif var == 'solar_elevation': 
            # Question for Joan: what is the unit of what this is returning? I'm gunna assume its degrees and I should just scale by 90
            ts = np.array(abs_time, dtype='datetime64[s]')
            out[:,i] = get_altitude_fast(x[:,in_varlist.index('lat_deg')], x[:,in_varlist.index('lon_deg')], ts) / 90. #note: this is supposed to be in lon /lat degrees 
        elif var.startswith('unnorm_'): assert False, "Data should be normalized before this"
        elif var == 'reltime_hours': out[:,i] = x[:,in_varlist.index('reltime_hours')] if 'reltime_hours' in in_varlist else x[:,in_varlist.index('reltime_seconds')] / 3600
        elif var in ['lat_deg','lon_deg']:
            if var in in_varlist:
                out[:,i] = x[:,in_varlist.index(var)]
            else:
                out[:,i] = x[:,in_varlist.index(var.replace('_deg','_rad'))] * 180 / np.pi
        elif var == 'pres_hpa': out[:,i] = x[:,in_varlist.index('pres_pa')] / 100.
        elif var.endswith("_present?"): out[:,i] = np.logical_not(np.isnan(x[:,in_varlist.index(var[:-len("_present?")])]))
        elif var in mesh.vars:
            assert f'{var}_present?' in out_varlist, f"Missing {var}_present? for {var} in out_varlist"
            out[:,i] = np.nan_to_num(x[:,in_varlist.index(var)],nan=0.0)
        elif var.startswith('onehot_'):
            varname = var.split('_')[1]
            varidx = int(var.split('_')[2])
            out[:,i] = x[:,in_varlist.index(varname)] == varidx
        elif var.startswith('sin(') or var.startswith('cos('):
                op = eval('np.'+var.split('(')[0])
                var = var.split('(')[1].split(')')[0]
                if '_deg' in var:
                    f = np.pi / 180
                else:
                    assert '_rad' in var
                    f = 1
                out[:,i] = op(x[:,in_varlist.index(var)]*f)
        else:
            assert False, "unknown variable to output: "+var
    assert not np.isnan(out).any(), f"nan in transformed data {out.shape} {np.sum(np.isnan(out))} {np.where(np.isnan(out))} {out_varlist[list(set(np.where(np.isnan(out))[1]))]}"
    return out

def reject_outliers(x,varlist,mesh,thresh=10):
    for i,var in enumerate(mesh.vars):
        j = varlist.index(var)
        #nn = np.isnan(x[:,j]).sum()
        bad = np.abs(x[:,j]) > thresh
        x[:,j] = np.where(bad,np.nan,x[:,j])
        #nn2 = np.isnan(x[:,j]).sum()
        #print(var,"rejected",nn2-nn,"outliers", (nn2-nn)/x.shape[0]*100,"%", "for mesh", mesh.string_id)
    return x


class ObsDataset():
    def __init__(self, mesh, is_required=False):
        self.mesh = mesh
        self.load_locations = ["/fast/proc/"]
        self.all_var_files = ["base"]
        self.ens_num = None
        self.is_required = is_required

    """
    def __getattr__(self, attr):
        if attr == '__getstate__': raise AttributeError
        if hasattr(self.mesh, attr): return getattr(self.mesh, attr)
        raise KeyError("uhh"+attr)
    """
    
    # Should be a function that just returns all the times that can be loaded from dataset
    # should return in unix time
    def get_loadable_times(self, datemin, datemax):
        times = self.get_file_times(datemin,datemax)
        alltimes = np.array([],dtype=np.int64)
        for dh in range(-self.mesh.da_window_size//2 + 1, self.mesh.da_window_size//2 + 1):
            tp = times - int(dh)*3600
            alltimes = np.union1d(tp,alltimes)
        return alltimes

    
    def load_data(self, nix):
        tots = []
        for dh in range(-self.mesh.da_window_size//2 + 1, self.mesh.da_window_size//2+1):

            data = self.load_hour(nix + dh*3600)

            if data is not None:
                time = data[:, self.mesh.full_varlist.index('reltime_hours')]
                data[:, self.mesh.full_varlist.index('reltime_hours')] = (time + dh) 

            tots.append(data)

        if all(x is None for x in tots):
            return torch.tensor([], dtype=torch.float16)
        ret = torch.cat([x for x in tots if x is not None], dim=0)
        N = ret.shape[0]
        N_subset = self.mesh.encoder_subset
        N_subset = min(N_subset, N)
        if N_subset != N:
            # get a random subset so it fits in ram
            perm = torch.randperm(N)[:N_subset]
            ret = ret[perm]
        return ret
    
    def summary_str(self):
        return"      [TODO: add metadata] <font color=\"green\">ObsLoader for mesh "+ self.mesh.__class__.__name__+ "</font>"

class MicrowaveDataset(ObsDataset):
    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        self.mesh = mesh
        self.source = 'satobs/'+mesh.instrument

    def get_file_times(self, datemin, datemax, load_location=None, extra=None):
        base = f'/fast/proc/{self.source}'
        yymm = date_range(datemin-timedelta(days=31),datemax+timedelta(days=31),freq='ME').strftime("%Y%m")
        tots = np.sort(np.concat([partial(seek,ext='.npz',fn=int)(f'{base}/{ym}') for ym in yymm]))
        return tots
    
    def load_hour(self, nix):
        mesh = self.mesh

        d = datetime(1970,1,1)+timedelta(seconds=int(nix))

        p = "/fast/proc/%s/%04d%02d/%d.npz" % (self.source, d.year, d.month, nix)
        if not os.path.exists(p):
            return

        f = np.load(p)['x']
        f[:, mesh.disk_varlist.index('said')] = [mesh.sats.index(int(a)) for a in f[:, mesh.disk_varlist.index('said')]]
        f = reject_outliers(f, mesh.disk_varlist, mesh, thresh=10)
        #f[:, 6:] = np.clip(f[:, 6:], -10., 10.)

        x = transform_data(f,nix,mesh.disk_varlist,mesh.full_varlist,mesh)
        return torch.from_numpy(x)

        out = np.zeros((f.shape[0], len(mesh.full_varlist) + 2), dtype=np.float16) + np.nan

        out[:, 0] = f[:, 0] # time # TODO: convert to +/- 3h instead of hourly
        out[:, -2] = f[:, 2] # lat
        out[:, -1] = f[:, 3] # lon

        def to_sincos(x, y):
            sx, cx, sy, cy = np.sin(x), np.cos(x), np.sin(y), np.cos(y)
            return np.array([sx, cx, sy, cy]).T
        mx_sats = len(mesh.sats)

        out[:, 1:5] = to_sincos(f[:, 2], f[:, 3])
        out[:, 5:9] = to_sincos(f[:, 4], f[:, 5])
        out[:, 9:9+mx_sats] = np.eye(mx_sats)[[mesh.sats.index(int(a)) for a in f[:, 1]]]
        nanmask = np.isnan(f[:, 6:])
        out[:, 9+mx_sats:9+mx_sats+mesh.nchan] = np.nan_to_num(f[:, 6:], copy=False, nan=0.0)
        idx = 9+mx_sats+mesh.nchan
        out[:, idx:idx+mesh.nchan] = (~nanmask)
        assert np.sum(np.isnan(out)) == 0

        return torch.from_numpy(out)


class WindborneDataset(ObsDataset):
    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        assert isinstance(mesh, WindborneData), "RadiosondeLoader is only compatible with RadiosondeData"
        self.mesh = mesh
        self.source = 'windborne'
        
        with open(f'/fast/proc/{self.source}/meta.json','r') as f:
            disk_varlist = json.load(f)['vars']
        assert disk_varlist == mesh.disk_varlist
            
        self.setup_norm(mesh)
    
    def setup_norm(self,mesh):
        norms = json.load(open('norm/norm.json','r'))
        mapping = {'temp':'130_t','rh':'rhlol','ucomp':'131_u','vcomp':'132_v','sh': '133_q'}
        norm_stds = np.zeros((len(mapping),len(norms['levels'])))
        norm_means = np.zeros((len(mapping),len(norms['levels'])))
        xl = np.array(norms['levels'])
        for i,k in enumerate(mesh.vars):
            ystd = np.array(norms[mapping[k]]['std'])
            ymean = np.array(norms[mapping[k]]['mean'])
            norm_stds[i,:] = np.interp(xl,xl[~np.isnan(ystd)],ystd[~np.isnan(ystd)])
            norm_means[i,:] = np.interp(xl,xl[~np.isnan(ymean)],ymean[~np.isnan(ymean)])
        self.norm_stds = norm_stds
        self.norm_means = norm_means
        self.norm_levels = xl  
    
    def normalize(self, x):
        norm_stds = self.norm_stds
        norm_means = self.norm_means
        norm_levels = self.norm_levels
        lev = x[:,self.mesh.disk_varlist.index('pres_pa')] / 100. # to hpa
        new_varlist = self.mesh.disk_varlist.copy()
        for i,var in enumerate(self.mesh.disk_varlist):
            if var.startswith("unnorm_"):
                nvar = var[7:].split("_")[0] # if it has a unit at the end, we remove it after normalization
                j = self.mesh.vars.index(nvar)
                scale = np.interp(lev,norm_levels,norm_stds[j])
                offset = np.interp(lev,norm_levels,norm_means[j])
                x[:,i] = (x[:,i] - offset) / scale
                new_varlist[i] = nvar
        return x, new_varlist
    
    def load_hour(self, nix):
        date = get_date(nix)
        path = "/fast/proc/%s/%04d%02d/%04d%02d%02d%02d.npz" % (self.source, date.year, date.month, date.year, date.month, date.day, date.hour)
        if not os.path.exists(path): return

        data = np.load(path)['x']
        data, curr_varlist = self.normalize(data)
        data = reject_outliers(data, curr_varlist, self.mesh, thresh=10)
        data = transform_data(data, nix, curr_varlist, self.mesh.full_varlist, self.mesh)
        data = torch.from_numpy(data)
        assert data.dtype == torch.float32
        assert not torch.any(torch.isnan(data)), "nan in radiosonde data"
        return data

    def get_file_times(self, datemin, datemax, load_location=None, extra=None):
        base = f'/fast/proc/{self.source}'
        yymm = date_range(datemin-timedelta(days=31),datemax+timedelta(days=31),freq='ME').strftime("%Y%m")
        tots = np.sort(np.concat([seek(f'{base}/{ym}', ext='.npz') for ym in yymm]))
        tots = tots[np.where(np.logical_and(to_unix(datemin) <= tots, tots <= to_unix(datemax)))]
        return tots


class RadiosondeDataset(ObsDataset):
    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        assert isinstance(mesh, RadiosondeData), "RadiosondeLoader is only compatible with RadiosondeData"
        self.mesh = mesh
        self.source = 'satobs/adpupa'
        
        with open(f'/fast/proc/{self.source}/meta.json','r') as f:
            disk_varlist = json.load(f)['vars']
        assert disk_varlist == mesh.disk_varlist
            
        self.setup_norm(mesh)
    
    def setup_norm(self,mesh):
        norms = json.load(open('norm/norm.json','r'))
        mapping = {'gp':'129_z','temp':'130_t','rh':'rhlol','ucomp':'131_u','vcomp':'132_v'}
        norm_stds = np.zeros((len(mapping),len(norms['levels'])))
        norm_means = np.zeros((len(mapping),len(norms['levels'])))
        xl = np.array(norms['levels'])
        for i,k in enumerate(mesh.vars):
            ystd = np.array(norms[mapping[k]]['std'])
            ymean = np.array(norms[mapping[k]]['mean'])
            norm_stds[i,:] = np.interp(xl,xl[~np.isnan(ystd)],ystd[~np.isnan(ystd)])
            norm_means[i,:] = np.interp(xl,xl[~np.isnan(ymean)],ymean[~np.isnan(ymean)])
        self.norm_stds = norm_stds
        self.norm_means = norm_means
        self.norm_levels = xl  
    
    def normalize(self, x):
        norm_stds = self.norm_stds
        norm_means = self.norm_means
        norm_levels = self.norm_levels
        lev = x[:,self.mesh.disk_varlist.index('pres_pa')] / 100. # to hpa
        new_varlist = self.mesh.disk_varlist.copy()
        for i,var in enumerate(self.mesh.disk_varlist):
            if var.startswith("unnorm_"):
                nvar = var[7:].split("_")[0] # if it has a unit at the end, we remove it after normalization
                j = self.mesh.vars.index(nvar)
                scale = np.interp(lev,norm_levels,norm_stds[j])
                offset = np.interp(lev,norm_levels,norm_means[j])
                x[:,i] = (x[:,i] - offset) / scale
                new_varlist[i] = nvar
        return x, new_varlist
    
    def load_hour(self, nix):
        date = get_date(nix)
        path = "/fast/proc/%s/%04d%02d/%04d%02d%02d%02d.npz" % (self.source, date.year, date.month, date.year, date.month, date.day, date.hour)
        if not os.path.exists(path): return

        data = np.load(path)['x']
        data, curr_varlist = self.normalize(data)
        data = reject_outliers(data, curr_varlist, self.mesh, thresh=5)
        data = transform_data(data, nix, curr_varlist, self.mesh.full_varlist, self.mesh)
        data = torch.from_numpy(data)
        assert data.dtype == torch.float32
        assert not torch.any(torch.isnan(data)), "nan in radiosonde data"
        return data

    def get_file_times(self, datemin, datemax, load_location=None, extra=None):
        base = f'/fast/proc/{self.source}'
        yymm = date_range(datemin-timedelta(days=31),datemax+timedelta(days=31),freq='ME').strftime("%Y%m")
        tots = np.sort(np.concat([seek(f'{base}/{ym}', ext='.npz') for ym in yymm]))
        tots = tots[np.where(np.logical_and(to_unix(datemin) <= tots, tots <= to_unix(datemax)))]
        return tots

class SatwindDataset(ObsDataset):
    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        assert isinstance(mesh, SatwindData), "SatwindLoader is only compatible with SatwindData"
        self.mesh = mesh
        self.source = 'satobs/satwnd'
        
        with open(f'/fast/proc/{self.source}/meta.json','r') as f:
            disk_varlist = json.load(f)['vars']
        assert disk_varlist == mesh.disk_varlist
            
        self.setup_norm(mesh)
    
    def setup_norm(self,mesh):
        norms = json.load(open('norm/norm.json','r'))
        mapping = {'gp':'129_z','temp':'130_t','rh':'rhlol','ucomp':'131_u','vcomp':'132_v'}
        norm_stds = np.zeros((len(mapping),len(norms['levels'])))
        norm_means = np.zeros((len(mapping),len(norms['levels'])))
        xl = np.array(norms['levels'])
        for i,k in enumerate(mesh.vars):
            ystd = np.array(norms[mapping[k]]['std'])
            ymean = np.array(norms[mapping[k]]['mean'])
            norm_stds[i,:] = np.interp(xl,xl[~np.isnan(ystd)],ystd[~np.isnan(ystd)])
            norm_means[i,:] = np.interp(xl,xl[~np.isnan(ymean)],ymean[~np.isnan(ymean)])
        self.norm_stds = norm_stds
        self.norm_means = norm_means
        self.norm_levels = xl  
    
    def normalize(self, x):
        norm_stds = self.norm_stds
        norm_means = self.norm_means
        norm_levels = self.norm_levels
        lev = x[:,self.mesh.disk_varlist.index('pres_pa')] / 100. # to hpa
        new_varlist = self.mesh.disk_varlist.copy()
        for i,var in enumerate(self.mesh.disk_varlist):
            if var.startswith("unnorm_"):
                nvar = var[7:].split("_")[0] # if it has a unit at the end, we remove it after normalization
                j = self.mesh.vars.index(nvar)
                scale = np.interp(lev,norm_levels,norm_stds[j])
                offset = np.interp(lev,norm_levels,norm_means[j])
                x[:,i] = (x[:,i] - offset) / scale
                new_varlist[i] = nvar
        return x, new_varlist
    
    def load_hour(self, nix):
        date = get_date(nix)
        path = "/fast/proc/%s/%04d%02d/%04d%02d%02d%02d.npz" % (self.source, date.year, date.month, date.year, date.month, date.day, date.hour)
        if not os.path.exists(path): return

        data = np.load(path)['x']
        data, curr_varlist = self.normalize(data)
        data = reject_outliers(data, curr_varlist, self.mesh, thresh=5)
        data = transform_data(data, nix, curr_varlist, self.mesh.full_varlist, self.mesh)
        data = torch.from_numpy(data)
        assert data.dtype == torch.float32
        assert not torch.any(torch.isnan(data)), "nan in satwind data"
        return data

    def get_file_times(self, datemin, datemax, load_location=None, extra=None):
        base = f'/fast/proc/{self.source}'
        yymm = date_range(datemin-timedelta(days=31),datemax+timedelta(days=31),freq='ME').strftime("%Y%m")
        tots = np.sort(np.concat([seek(f'{base}/{ym}', ext='.npz') for ym in yymm]))
        tots = tots[np.where(np.logical_and(to_unix(datemin) <= tots, tots <= to_unix(datemax)))]
        return tots
    
class SurfaceDataset(ObsDataset):
    def __init__(self,mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        assert isinstance(mesh, SurfaceData), "IgraLoader is only compatible with SurfaceData"
        self.mesh = mesh
        self.setup_norm(mesh)
        self.source = 'sfc'
    
    def setup_norm(self,mesh):
        norms = json.load(open('norm/norm.json','r'))
        norms['elev'] = {'mean':0,'std':3000.}
        norms['is_ship'] = {'mean':0,'std':1} # lol
        mapping = {'mslp': '151_msl', '2t': '167_2t', '2d': '168_2d', '10u': '165_10u', '10v': '166_10v', 'elev': 'elev', 'is_ship': 'is_ship'}
        norm_stds = np.zeros(len(mapping))
        norm_means = np.zeros(len(mapping))
        for i,k in enumerate(mesh.vars):
            ystd = np.array(norms[mapping[k]]['std'])
            ymean = np.array(norms[mapping[k]]['mean'])
            norm_stds[i] = ystd
            norm_means[i] = ymean
        self.norm_stds = norm_stds
        self.norm_means = norm_means

        with open('/fast/proc/satobs/sfc/meta.json','r') as f:
            self.disk_varlist = json.load(f)['vars']

        assert self.disk_varlist == self.mesh.disk_varlist

    def normalize(self,x):
        norm_stds = self.norm_stds
        norm_means = self.norm_means
        new_varlist = self.mesh.disk_varlist.copy()
        for i,var in enumerate(self.mesh.disk_varlist):
            if var.startswith("unnorm_") and "viz" not in var and "pr_mm" not in var:
                nvar = var[7:].split("_")[0] # if it has a unit at the end, we remove it after normalization
                j = self.mesh.vars.index(nvar)
                scale = norm_stds[j]
                offset = norm_means[j]
                if "_hpa" in var:
                    scale /= 100.
                    offset /= 100.
                if var.endswith("_c"):
                    offset -= 273.15
                x[:,i] = (x[:,i] - offset) / scale
                new_varlist[i] = nvar
        return x, new_varlist
    
    def load_hour(self, nix):
        mesh = self.mesh
        d = get_date(nix)
        p = "/fast/proc/satobs/sfc/%04d%02d/%04d%02d%02d%02d.npy" % (d.year, d.month, d.year, d.month, d.day, d.hour)
        if not os.path.exists(p):
            return
        
        data = np.load(p)
        x, curr_varlist = self.normalize(data)
        x = reject_outliers(x,curr_varlist,mesh,thresh=10)
        x = transform_data(x,nix,curr_varlist,mesh.full_varlist,mesh)
        x = torch.from_numpy(x)
        assert x.dtype == torch.float32
        assert not torch.any(torch.isnan(x)), "nan in station data"
        return x
    


    def get_file_times(self, datemin, datemax, load_location=None, extra=None):
        base = f'/fast/proc/satobs/sfc'
        return IgraDataset._get_file_times(base,datemin,datemax)

    @cache.memoize(expire=60*60*24)
    @staticmethod
    def _get_file_times(base,datemin,datemax):
        yymm = date_range(datemin-timedelta(days=31),datemax+timedelta(days=31),freq='ME').strftime("%Y%m")
        tots = np.sort(np.concat([seek(f'{base}/{ym}') for ym in yymm]))
        tots = tots[np.where(np.logical_and(to_unix(datemin) <= tots, tots <= to_unix(datemax)))]
        return tots

    def print_stats(self,x):
        print(f"Total points: {x.shape[0]}")
        print(f"Unique soundings: {torch.unique(x[:,0]).shape[0]}")

class IgraDataset(ObsDataset):
    def __init__(self,mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        assert isinstance(mesh, BalloonData), "IgraLoader is only compatible with BalloonData"
        self.mesh = mesh
        self.setup_norm(mesh)
        self.source = 'igra'
    
    def setup_norm(self,mesh):
        norms = json.load(open('norm/norm.json','r'))
        mapping = {'gp':'129_z','temp':'130_t','rh':'rhlol','ucomp':'131_u','vcomp':'132_v'}
        norm_stds = np.zeros((len(mapping),len(norms['levels'])))
        norm_means = np.zeros((len(mapping),len(norms['levels'])))
        xl = np.array(norms['levels'])
        for i,k in enumerate(mesh.vars):
            ystd = np.array(norms[mapping[k]]['std'])
            ymean = np.array(norms[mapping[k]]['mean'])
            norm_stds[i,:] = np.interp(xl,xl[~np.isnan(ystd)],ystd[~np.isnan(ystd)])
            norm_means[i,:] = np.interp(xl,xl[~np.isnan(ymean)],ymean[~np.isnan(ymean)])
        self.norm_stds = norm_stds
        self.norm_means = norm_means
        self.norm_levels = xl   

        with open('/fast/proc/igra/meta.json','r') as f:
            self.disk_varlist = json.load(f)['vars']

        assert self.disk_varlist == self.mesh.disk_varlist

    def normalize(self,x):
        norm_stds = self.norm_stds
        norm_means = self.norm_means
        norm_levels = self.norm_levels
        lev = x[:,self.mesh.disk_varlist.index('pres_pa')] / 100. # to hpa
        new_varlist = self.mesh.disk_varlist.copy()
        for i,var in enumerate(self.mesh.disk_varlist):
            if var.startswith("unnorm_"):
                nvar = var[7:].split("_")[0] # if it has a unit at the end, we remove it after normalization
                j = self.mesh.vars.index(nvar)
                scale = np.interp(lev,norm_levels,norm_stds[j])
                offset = np.interp(lev,norm_levels,norm_means[j])
                x[:,i] = (x[:,i] - offset) / scale
                new_varlist[i] = nvar
        return x, new_varlist
    
    def load_hour(self, nix):
        mesh = self.mesh
        d = get_date(nix)
        p = "/fast/proc/%s/%04d%02d/%04d%02d%02d%02d.npy" % (self.source, d.year, d.month, d.year, d.month, d.day, d.hour)
        if not os.path.exists(p):
            return
        
        data = np.load(p)
        x, curr_varlist = self.normalize(data)
        x = reject_outliers(x,curr_varlist,mesh,thresh=10)
        x = transform_data(x,nix,curr_varlist,mesh.full_varlist,mesh)
        x = torch.from_numpy(x)
        assert x.dtype == torch.float32
        assert not torch.any(torch.isnan(x)), "nan in igra data"
        return x
    


    def get_file_times(self, datemin, datemax, load_location=None, extra=None):
        base = f'/fast/proc/{self.source}'
        return IgraDataset._get_file_times(base,datemin,datemax)

    @cache.memoize(expire=60*60*24)
    @staticmethod
    def _get_file_times(base,datemin,datemax):
        yymm = date_range(datemin-timedelta(days=31),datemax+timedelta(days=31),freq='ME').strftime("%Y%m")
        tots = np.sort(np.concat([seek(f'{base}/{ym}') for ym in yymm]))
        tots = tots[np.where(np.logical_and(to_unix(datemin) <= tots, tots <= to_unix(datemax)))]
        return tots

    def print_stats(self,x):
        print(f"Total points: {x.shape[0]}")
        print(f"Unique soundings: {torch.unique(x[:,0]).shape[0]}")

    def plot_igra_data_tensor(self,x,date):
        os.makedirs('ignored/igra/',exist_ok=True)
        mesh = self.mesh
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)    
        x = x.detach().cpu().numpy()
        lats = x[:,3]; lons = x[:,4]
        ax.scatter(lons, lats, s=10, transform=ccrs.PlateCarree())
        ax.plot(lons,lats,alpha=0.1)
        plt.tight_layout()
        plt.savefig(f'ignored/igra/{date.strftime("%Y-%m-%d-%H")}_map.png')
        plt.title(f"{date.strftime('%Y-%m-%d %H')}z")
        plt.close()
        fig,ax = plt.subplots(len(mesh.full_varlist),1,figsize=(30,45))
        for i,v in enumerate(mesh.full_varlist):
            ax[i].plot(x[:,i])
            ax[i].set_title(v); ax[i].grid()
        plt.tight_layout(); plt.savefig(f'ignored/igra/{date.strftime("%Y-%m-%d-%H")}_timeseries.png')

class StationDataset():
    def __init__(
            self, 
            mesh: StationData,
            batch_size: int = 192,
            metar_frac: float = 0.9,
            validation: bool = False,
            is_required: bool = True,
        ):
        """
        TODO: implement validation split
        """
        assert isinstance(mesh, StationData), "StationLoader is only compatible with StationData"
        self.mesh = mesh
        self.metar_location = '/fast/proc/hres_consolidated/consolidated'
        self.madis_location = '/fast/ignored/merged'
        self.load_locations = [self.metar_location, self.madis_location]
        self.all_var_files = ["base"] # TODO: not sure what this is
        self.source = 'station' # TODO: also not sure what this is
        self.batch_size = batch_size
        self.metar_frac = metar_frac
        self.validation = validation
        self.half_window_seconds = 10 * 60 # obs are between time - 10min and time + 10min
        self.is_required = is_required
        self.setup_norm()
        self.setup_rounding_weights()
    
    def setup_norm(self):
        norm = json.load(open('/fast/haoxing/deep/norm/norm.json','r'))
        norm_means, norm_stds = [], []
        for var in self.mesh.vars:
            ncar_var = self.mesh.metar_to_ncar[var]
            norm_means.append(norm[ncar_var]['mean'])
            norm_stds.append(norm[ncar_var]['std'])
        self.norm_means = np.array(norm_means)
        self.norm_stds = np.array(norm_stds)
    
    def setup_rounding_weights(self):
        """I (haoxing) think rounding weights are based on how precise the station location is?
           Actually not sure lol"""
        rounding_path = "/fast/proc/hres_consolidated/consolidated/station_rounding"
        avail_years = os.listdir(rounding_path)
        avail_years = [int(x[:4]) for x in avail_years]
        self.rounding_weights = {}
        for year in avail_years:
            with open(f"{rounding_path}/{year}.pkl", "rb") as f:
                rounding = pickle.load(f)
                self.rounding_weights[year] = np.concatenate([
                    np.zeros(self.mesh.n_madis) + 0.1,
                    rounding
                ])

    def process_metar(self, metar: np.ndarray, dt: datetime):
        """Copied/rewritten from HresDataset"""
        ts = metar[:,0]
        start = to_unix(dt - timedelta(seconds=self.half_window_seconds))
        end = to_unix(dt + timedelta(seconds=self.half_window_seconds))
        metar = metar[(ts >= start) & (ts <= end), :]
        metar[:, 0] -= (dt - datetime(1970, 1, 1)).total_seconds() # relative to dt
        metar[:, 1] += self.mesh.n_madis # madis is first
        metar[:, 2] /= 100. # hPa to match madis
        metar[:, 3] -= 273.15 # C to match madis
        metar[:, 4] -= 273.15 # C to match madis
        ucomp = metar[:, 5].copy()
        vcomp = metar[:, 6].copy()
        metar[:, 5] = vcomp
        metar[:, 6] = ucomp
        metar[:, 7] *= 0.3048 # ft -> m
        metar[:, 8] *= 1.60934 # miles -> km
        metar = np.hstack((metar, np.zeros((metar.shape[0], 1), dtype=np.float32)+np.nan)) # pad with nan for precip
        return metar

    def index(self, data, col):
        return data[:, self.mesh.COLUMNS.index(col)]

    def compute_sample_weights(self, dt: datetime, stations: np.ndarray):
        if dt.year in self.rounding_weights:
            perc_rounded = self.rounding_weights[dt.year][stations]
            is_imprecise = perc_rounded > 0.5
            is_imprecise += perc_rounded < 0.025
            is_imprecise = 2 * is_imprecise
        else:
            is_imprecise = np.zeros_like(stations)

        def is_fuzzy(point):
            lat, lon = point
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return np.inf
            if round(lat, 2) == round(lat, 3) and round(lon, 2) == round(lon, 3):
                return 2
            else:
                return 0
        
        is_fuzzyloc = np.array([is_fuzzy(self.mesh.all_latlons[station]) for station in stations])
        is_slop = np.array([5 if self.mesh.is_slop[station] else 0 for station in stations])

        return 1/(1 + is_fuzzyloc + is_slop + is_imprecise)

    def load_data_old(self, ts: int):
        dt = datetime.utcfromtimestamp(ts)
        metar_file = f"{self.metar_location}/{dt.year}/{dt.month:02}{dt.day:02}.npy"
        madis_file = f"{self.madis_location}/{dt.year}/{dt.month:02}/{dt.year}{dt.month:02}{dt.day:02}{dt.hour:02}.npz"
        if not os.path.exists(metar_file) or not os.path.exists(madis_file):
            return None
        madis = np.load(madis_file)["data"]
        metar = np.load(metar_file)
        metar = self.process_metar(metar, dt)
        data = np.concatenate((madis, metar), axis=0)

        out = np.empty((data.shape[0], len(self.mesh.vars)), dtype=np.float32)
        for i, var in enumerate(self.mesh.vars):
            out[:, i] = self.index(data, var)
            if var in ["tmpf", "dwpf"]:
                out[:, i] += 273.15
            if var == "mslp":
                out[:, i] *= 100.
            if var == "logprecip":
                out[:, i] = np.log(np.maximum(out[:, i] + 1e-7, 0))
            out[:, i] = (out[:, i] - self.norm_means[i]) / self.norm_stds[i]
        out = torch.from_numpy(out)

        stations = self.index(data, "station").astype(np.int32)
        is_slop = np.array([5 if self.mesh.is_slop[station] else 0 for station in stations])
        # remove slop winds cuz they're bad somehow
        out[is_slop != 0, 3] = np.nan
        out[is_slop != 0, 4] = np.nan
        weights = self.compute_sample_weights(dt, stations)
        
        # select a random subset of stations according to metar_frac
        n_metar_dt = metar.shape[0]
        n_madis_dt = madis.shape[0]
        metar_frac = max(self.metar_frac, n_metar_dt / (n_metar_dt + n_madis_dt))
        n_metar_batch = min(int(self.batch_size * metar_frac), n_metar_dt)
        # enforce metar_frac and underfill batch if needed
        n_madis_batch = min(
            self.batch_size - n_metar_batch,
            int(n_metar_batch / metar_frac * (1 - metar_frac))
        )
        
        madis_idxs = np.random.choice(n_madis_dt, size=n_madis_batch, replace=False) if n_madis_batch < n_madis_dt else np.arange(n_madis_dt)
        metar_idxs = np.random.choice(range(n_madis_dt, n_madis_dt + n_metar_dt), size=n_metar_batch, replace=False) if n_metar_batch < n_metar_dt else np.arange(n_madis_dt, n_madis_dt + n_metar_dt)
        batch_idxs = np.concatenate((madis_idxs, metar_idxs))
        batch_data, batch_weights = out[batch_idxs], weights[batch_idxs]
        batch_stations = stations[batch_idxs]
        batch_latlons = torch.tensor([self.mesh.all_latlons[station] for station in batch_stations])
        return batch_latlons, batch_data, batch_weights
    
    def load_data(self, ts: int):
        dt = datetime.utcfromtimestamp(ts)
        metar_file = f"{self.metar_location}/{dt.year}/{dt.month:02}{dt.day:02}.npy"
        madis_file = f"{self.madis_location}/{dt.year}/{dt.month:02}/{dt.year}{dt.month:02}{dt.day:02}{dt.hour:02}.npz"
        if not os.path.exists(metar_file) or not os.path.exists(madis_file):
            return None
        madis = np.load(madis_file)["data"]
        metar = np.load(metar_file)
        metar = self.process_metar(metar, dt)
        data = np.concatenate((madis, metar), axis=0)

        out = np.empty((data.shape[0], len(self.mesh.vars)), dtype=np.float32)
        for i, var in enumerate(self.mesh.vars):
            out[:, i] = self.index(data, var)
            if var in ["tmpf", "dwpf"]:
                out[:, i] += 273.15
            if var == "mslp":
                out[:, i] *= 100.
            if var == "logprecip":
                out[:, i] = np.log(np.maximum(out[:, i] + 1e-7, 0))
            out[:, i] = (out[:, i] - self.norm_means[i]) / self.norm_stds[i]

        stations = self.index(data, "station").astype(np.int32)
        is_slop = np.array([5 if self.mesh.is_slop[station] else 0 for station in stations])
        # remove slop winds cuz they're bad somehow
        out[is_slop != 0, 3] = np.nan
        out[is_slop != 0, 4] = np.nan
        weights = self.compute_sample_weights(dt, stations)
        
        # select a random subset of stations according to metar_frac
        n_metar_dt = metar.shape[0]
        n_madis_dt = madis.shape[0]
        metar_frac = max(self.metar_frac, n_metar_dt / (n_metar_dt + n_madis_dt))
        n_metar_batch = min(int(self.batch_size * metar_frac), n_metar_dt)
        # enforce metar_frac and underfill batch if needed
        n_madis_batch = min(
            self.batch_size - n_metar_batch,
            int(n_metar_batch / metar_frac * (1 - metar_frac))
        )

        B = self.batch_size

        dt_latlons = self.mesh.all_latlons[stations]

        centers = self.index(metar[torch.randperm(n_metar_dt)[:B]],'station').astype(np.int32)
        centers = [self.mesh.all_latlons[center] for center in centers]
        batch_data = []
        for center in centers:
            center += np.random.rand(2) - 0.5   # we shift by a radom amount so that the model doesn't primarly learn to predict
                                                # points in the center of the high res patch
            is_close = np.all(np.abs(dt_latlons - center) < 0.5,axis=-1)
            close_idx = np.where(is_close)[0]
            close = {'center': center, 'latlons': dt_latlons[close_idx], 'data': out[close_idx], 'weights': weights[close_idx]}
            close = {k: torch.from_numpy(v).to(torch.float32) for k, v in close.items()}
            batch_data.append(close)

        return batch_data


    def __getitem__(self, idx: int):
        dt = self.datetimes[idx]
        return self.load_data(dt)
    
    def get_loadable_times(self, datemin, datemax, load_location=None, extra=None):
        base = self.metar_location
        yyyys = date_range(datemin-timedelta(days=31),datemax+timedelta(days=31),freq='YE').strftime("%Y")
        # the files look like
        # /path/2015/0101.npy
        def seek(yyyy):
            dates = list(map(lambda p: datetime.strptime(f"{yyyy}{p.stem}", "%Y%m%d"), pathlib.Path(f'{base}/{yyyy}').rglob('[0-9]*.npy')))
            timestamps = []
            for date in dates:
                for hour in range(24):
                    timestamps.append(to_unix(date + timedelta(hours=hour)))
            return np.array(timestamps, dtype=np.int64)
        tots = np.sort(np.concatenate(list(map(seek,yyyys))))
        tots = tots[np.where(np.logical_and(to_unix(datemin) <= tots, tots <= to_unix(datemax)))]
        return tots

    def summary_str(self):
        return self.mesh.summary_str()

if __name__ == "__main__":
    loader = SurfaceDataset(SurfaceData())
    for i in range(10):
        x = loader.load_hour(to_unix(datetime(2021, 1, 1, i)))
        import pdb; pdb.set_trace()
    exit()
    start = datetime(2015,1,1)
    end = datetime(2015,1,10)
    d = start
    while d < end:
        d += timedelta(hours=1)
        print(f"## Data for {d}")
        path = f'/fast/proc/igra/{d.strftime("%Y%m")}/{d.strftime("%Y%m%d%H")}.npy'
        x = loader.load_data(path)
        if x is None: continue
        loader.print_stats(x)
        loader.plot_igra_data_tensor(x,d)

if __name__ == "__main__":
    ds = StationDataset(StationData())
    ds.load_data(datetime(2015, 1, 1, 0))
    loader = IgraDataset(BalloonData())
    start = datetime(2015,1,1)
    end = datetime(2015,1,10)
    d = start
    while d < end:
        d += timedelta(hours=1)
        print(f"## Data for {d}")
        path = f'/fast/proc/igra/{d.strftime("%Y%m")}/{d.strftime("%Y%m%d%H")}.npy'
        x = loader.load_data(path)
        if x is None: continue
        loader.print_stats(x)
        loader.plot_igra_data_tensor(x,d)
