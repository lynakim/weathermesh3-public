import copy
from datetime import timedelta, datetime
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import os
import pickle
import time
import torch
from types import SimpleNamespace
import xarray as xr
import zipfile

from gen1.utils import (
    CONSTS_PATH,
    PROC_PATH,
    RUNS_PATH,
    D,
    levels_full,
    levels_tiny,
    levels_ecm1,
    levels_ecm2,
    core_pressure_vars,
    core_sfc_vars,
    ncar2cloud_names,
    num2levels,
    dimprint,
    to_unix,
    get_date,
    get_date_str,
    get_dates,
    sdate2nix,
    load_state_norm,
    mapll,
    flattenll,
    interp_levels,
    set_metadata,
    TIMEIT,
    get_proc_path_base,
    get_proc_path_time,
    select_bbox,
    NeoDatasetConfig
)

def to_realtime(mesh):
    mesh = copy.deepcopy(mesh) # I think it's best to deepcopy the mesh but tbh not 100% sure
    match mesh.source:
        case 'neogfs-25': mesh.source = 'gfs_rt-25'
        case 'hres-13': 
            mesh.source = 'ens_rt-16'
            mesh.input_levels = levels_ecm1
            mesh.intermediate_levels = [levels_tiny]
    return mesh

class NeoDataConfig():
    def __init__(self,inputs=[],ouputs=[],conf_to_copy=None,**kwargs):
        self.inputs = inputs 
        self.outputs = ouputs
        self.num_workers = 2
        self.worker_complain = False 
        self.max_instance_queue_size = 8
        self.max_sample_queue_size = 1
        self.timesteps = [24]
        #self.requested_dates = get_dates((D(1997, 1, 1),D(2007, 1,1)))
        self.requested_dates = get_dates((D(1997, 1, 1),D(2017, 12,1)))
        self.only_at_z = None
        self.seed = None
        self.max_ram_manual = 6 * 2**30
        self.ram_margin = 1 * 2**30   # worker will fill to this amount above max, receiver will unload to this amount below max
        self.clamp = 13
        self.clamp_output=None
        self.additional_vars = ['rad','timeofday']
        self.num_times_use_factor = 60*60
        self.odds_idc = 0.05
        self.simple_mode = False
        self.world_size = None
        self.rank = 0
        self.log_dir = None
        self.log = True
        self.quiet = False
        self.cycle_ram = True
        self.rank = 0
        self.world_size = 1
        self.name = ''
        self.use_mmap = False
        self.mmap_cache_path = '/tmp/dataloader_mmap/'
        self.mmap_cache_size = 1*2**40
        self.realtime = False
        self.ens_nums = None

        if conf_to_copy is not None:
            self.__dict__.update(copy.deepcopy(conf_to_copy.__dict__))
        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a NeoDataConfig attribute"
            setattr(self,k,v)
        self.update()

    def update(self):
        if len(self.outputs) == 0:
            self.outputs = self.inputs
        self.proc_start_t = time.time()
        if self.max_ram_manual is not None:
            self.max_ram = self.max_ram_manual

        if self.clamp_output is None:
            self.clamp_output = self.clamp
        ts0 = [0] + self.timesteps
        #assert np.diff(ts0).min() == np.diff(ts0).max()
        assert ts0 == sorted(ts0)
        self.model_dh = np.diff(ts0)[0] if len(ts0) > 1 else None

        if self.seed is None:
            self.seed = int(np.modf(time.time()*10)[0]*1000)

        if self.realtime:
            for i,mesh in enumerate(self.inputs):
                self.inputs[i] = to_realtime(mesh)
                if self.ens_nums is not None:
                    self.inputs[i].ens_num = self.ens_nums[i]
        else:
            assert self.ens_nums is None, "Ensemble numbers are only supported in realtime mode"


Gzarr = None
def get_zarr():
    global Gzarr
    if Gzarr is None:
        print("loading zarr")
        #Gzarr = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2")
        Gzarr = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr")

        print("loaded zarr")
    return Gzarr

class WeatherInstance():
    def __init__(self,idx,config):
        self.USE_MMAP = config.use_mmap
        self.idx = idx
        self.date = config.instance_dates[idx]
        self.nix = to_unix(self.date)
        self.from_cache = False
        if self.USE_MMAP:
            mesh = config.instance_meshes[idx]
            os.makedirs(config.mmap_cache_path,exist_ok=True)
            self.path = WeatherInstance.get_path_str(config,mesh)

    @staticmethod
    def get_path_str(config : NeoDataConfig, mesh : NeoDatasetConfig):
        c : NeoDatasetConfig = mesh
        sts = f"{c.source}{''.join(c.pressure_vars)}{''.join(c.sfc_vars)}".replace("_","")
        for i in range(10): sts = sts.replace(str(i),"")
        path = config.mmap_cache_path + f"{c.subsamp}-{len(c.levels)}-{sts}."
        return path

    def write(self,pr,sfc):
        self.data_pr_meta = SimpleNamespace(shape=pr.shape,dtype=pr.dtype,nbytes=pr.nbytes)
        self.data_sfc_meta = SimpleNamespace(shape=sfc.shape,dtype=sfc.dtype,nbytes=sfc.nbytes)
        if self.USE_MMAP:
            self.write_mmap(pr,sfc)
        else:
            self.write_shm(pr,sfc)

    def write_mmap(self, pr, sfc):
        self.from_cache = False
        total_size = np.prod(pr.shape) + np.prod(sfc.shape)

        combined_path = self.path + f'{get_date_str(self.nix)}.bin'
        combined_mmap = np.memmap(combined_path, dtype=sfc.dtype, mode='w+', shape=(total_size,))

        pr_size = np.prod(pr.shape)
        pr_slice = slice(0, pr_size)
        sfc_slice = slice(pr_size, total_size)

        combined_mmap[pr_slice] = pr.ravel()
        combined_mmap[sfc_slice] = sfc.ravel()
        #combined_mmap.flush()
        #del combined_mmap

        self.data_pr_meta.path = combined_path
        self.data_sfc_meta.path = combined_path

        if not os.path.exists(self.path + "pickle"):
            with open(self.path + "pickle", "wb") as f:
                pickle.dump({"pr": self.data_pr_meta.__dict__, "sfc": self.data_sfc_meta.__dict__}, f)


    def load_cache(self):
        if not os.path.exists(self.path + "pickle") or not os.path.exists(self.path + f"{get_date_str(self.nix)}.bin"):
            return False
        try:
            with open(self.path + "pickle", "rb") as f:
                j = pickle.load(f)
            self.data_pr_meta = SimpleNamespace(**j["pr"])
            self.data_pr_meta.path = self.path + f"{get_date_str(self.nix)}.bin"
            self.data_sfc_meta = SimpleNamespace(**j["sfc"])
            self.data_sfc_meta.path = self.path + f"{get_date_str(self.nix)}.bin"
            self.from_cache = True
            #print(f'got from cache {get_date_str(self.nix)}' )
            return True
        except Exception as e:
            return False

    def write_shm(self,pr,sfc):
        assert not self.USE_MMAP
        pr_shm = SharedMemory(create=True, size=pr.nbytes)
        self.data_pr_meta.path = pr_shm.name 
        pr_mm = np.ndarray(pr.shape, dtype=pr.dtype, buffer=pr_shm.buf)
        pr_mm[:] = pr[:]

        sfc_shm = SharedMemory(create=True, size=sfc.nbytes)
        self.data_sfc_meta.path = sfc_shm.name
        sfc_mm = np.ndarray(sfc.shape, dtype=sfc.dtype, buffer=sfc_shm.buf)
        sfc_mm[:] = sfc[:]

        pr_shm.close()
        sfc_shm.close()

        del pr_mm
        del sfc_mm
        del pr_shm
        del sfc_shm

    def load(self):
        if self.USE_MMAP:
            combined_path = self.data_pr_meta.path  # or self.data_sfc_meta.path, both should be the same
            #print("loading", combined_path)
            assert get_date_str(self.nix) in combined_path
            try: 
                combined_mmap = np.memmap(combined_path, dtype=self.data_sfc_meta.dtype, mode='readwrite', shape=(np.prod(self.data_pr_meta.shape) + np.prod(self.data_sfc_meta.shape),))
                #combined_mmap = np.load(combined_path)
                #with open(combined_path, 'rb') as f:
                #    data = f.read()
                #    combined_mmap = np.frombuffer(data, dtype=self.data_sfc_meta.dtype).reshape((np.prod(self.data_pr_meta.shape) + np.prod(self.data_sfc_meta.shape),))

            except FileNotFoundError:
                print("Damn got unlucky")
                return False
            pr_size = np.prod(self.data_pr_meta.shape)
            self.data_pr = np.reshape(combined_mmap[:pr_size], self.data_pr_meta.shape)
            self.data_sfc = np.reshape(combined_mmap[pr_size:], self.data_sfc_meta.shape)
            # Touch each element to force loading into RAM without duplication
            _ = self.data_pr[:1, :1]
            _ = self.data_sfc[:1, :1]
            #_ = np.sum(self.data_pr)
            #_ = np.sum(self.data_sfc)
            #print("TOUCHED")
        else:
            self.data_pr_shm = SharedMemory(name=self.data_pr_meta.path)
            self.data_sfc_shm = SharedMemory(name=self.data_sfc_meta.path)
            self.data_pr = np.ndarray(self.data_pr_meta.shape, dtype=self.data_pr_meta.dtype, buffer=self.data_pr_shm.buf)
            self.data_sfc = np.ndarray(self.data_sfc_meta.shape, dtype=self.data_sfc_meta.dtype, buffer=self.data_sfc_shm.buf)
        return True

    def cleanup(self):
        if self.USE_MMAP:
            pass # lol mmap is good
        else:
            self.data_pr_shm.close()
            self.data_pr_shm.unlink()
            self.data_sfc_shm.close()
            self.data_sfc_shm.unlink()

def load_from_npz(zf, name):
    # figure out offset of .npy in .npz
    info = zf.NameToInfo[name + '.npy']
    assert info.compress_type == 0
    zf.fp.seek(info.header_offset + len(info.FileHeader()) + 20)
    # read .npy header
    version = np.lib.format.read_magic(zf.fp)
    np.lib.format._check_version(version)
    shape, fortran_order, dtype = np.lib.format._read_array_header(zf.fp,
                                                                   version)
    offset = zf.fp.tell()
    # create memmap
    return np.memmap(zf.filename, dtype=dtype, shape=shape,
                     order='F' if fortran_order else 'C', mode='r',
                     offset=offset)

def neo_get_latlon_input(date, mesh):
    assert mesh.type == "latlon"
    cp = get_proc_path_time(date,mesh)
    assert os.path.exists(cp), cp
    assert mesh.subsamp == 1, "John on Jan7 wasn't sure if we ever use subsamp > 1"
    if mesh.parent is None: 
        cf = np.load(cp)
        pr,sfc = cf["pr"][:720], cf["sfc"][:720]
    else: 
        pmesh = mesh.parent
        assert pmesh.parent is None, "This is not yet supported"
        zf = zipfile.ZipFile(cp)
        pr = load_from_npz(zf, 'pr')
        sfc = load_from_npz(zf, 'sfc')
        zf.close()
        pr = select_bbox(pr,pmesh,mesh.bbox)
        sfc = select_bbox(sfc,pmesh,mesh.bbox)

    assert mesh.extra_pressure_vars == [], "This is not yet supported"
    #print(date, "making neo latlon input", mesh.extra_sfc_vars, mesh.extra_sfc_pad, core_sfc_vars)
    if len(mesh.extra_sfc_vars) > 0:
        assert mesh.parent is None, "This is not yet supported"
        sfc_total = np.zeros((sfc.shape[0],sfc.shape[1],len(mesh.sfc_vars)))
        sfc_total[:,:,:len(core_sfc_vars)] = sfc
        for i,e in enumerate(mesh.extra_sfc_vars):
            cp = get_proc_path_time(date,mesh,extra=e)
            assert os.path.exists(cp), cp
            cfe = np.load(cp)['x']
            sfc_total[:,:,len(core_sfc_vars)+i] = cfe # *0 + 0.666
        sfc = sfc_total
        #print("sfc shape", sfc.shape)
    return pr,sfc

FAKE_LOAD = False

class WeatherInstanceLocal(WeatherInstance):
    def __init__(self,idx,config):
        #dataset is for some cloud bullshat that may not even matter
        super().__init__(idx,config)
        if self.USE_MMAP:
            if self.load_cache():
                return
        pr, sfc = neo_get_latlon_input(self.date, config.instance_meshes[idx])
        self.write(pr,sfc)

    def get_size(self):
        if FAKE_LOAD:
            return 300*2**20
        return self.data_pr_meta.nbytes + self.data_sfc_meta.nbytes

class WeatherInstanceCloud(WeatherInstance):
    def __init__(self,idx,config):
        #assert False, "This probalby needs love before it works. THink about how the mesh interface changed. GL brother"
        super().__init__(idx,config)
        if self.USE_MMAP:
            if self.load_cache():
                return
        pr, sfc = self.get_data(idx, config)
        self.write(pr,sfc)

    def get_size(self):
        return self.data_pr_meta.nbytes + self.data_sfc_meta.nbytes
    
    @staticmethod
    def get_data(idx,config):
        xpr = []
        xsfc = []
        t0 = time.time()
        ccc = 0
        date = config.instance_dates[idx]
        #print(os.getpid(), "loading", date)
        nix = to_unix(date)
        cachepath = "/fast/weatherbench_cache/%04d/%02d/%d.npz" % (date.year, date.month, int(nix))
        if os.path.exists(cachepath):
            out = np.load(cachepath)
            return out["pr"], out["sfc"]
        os.makedirs(os.path.dirname(cachepath), exist_ok=True)
        dataset = get_zarr()
        data = dataset.sel(time=np.datetime64(nix, 's'))
        mesh = config.instance_meshes[idx]
        lol = []
        for w in mesh.pressure_vars:
            v = ncar2cloud_names[w]
            mean, std2 = config.norms[w]
            x = time.time()
            ohp = data[v][mesh.wh_lev, :720, :].transpose('latitude', 'longitude', 'level').to_numpy()
            ccc += (time.time()-x)
            ohp = (ohp - mean[np.newaxis, np.newaxis, mesh.wh_lev])/(std2[np.newaxis, np.newaxis, mesh.wh_lev])

            if np.max(np.abs(ohp)) > np.finfo(np.float16).max:
                ohp = np.minimum(np.abs(ohp), np.finfo(np.float16).max) * np.sign(ohp)
            lol.append(ohp.astype(np.float16))
            
        lol = np.array(lol).transpose(1, 2, 0, 3)
        xpr.append(lol)
        
        lol = []
        for w in mesh.sfc_vars:
            v = ncar2cloud_names[w]
        
            mean, std2 = config.norms[w]
            x = time.time()
            ohp = data[v][:720, :].to_numpy()
            if v == "total_precipitation":
                print('god',w,v)
                ohp = np.log(np.maximum(ohp + 1e-7,0))
            ccc += (time.time()-x)
            ohp = (ohp - mean[np.newaxis, np.newaxis])/(std2[np.newaxis, np.newaxis])
            lol.append(ohp.astype(np.float16)[0])
        lol = np.array(lol).transpose(1, 2, 0)
        xsfc.append(lol)
        print(f"cloud loading took total: {time.time()-t0:.2f}, inner: {ccc:0.2f}")
        cp = cachepath.replace(".npz", ".tmp.npz")
        np.savez(cp, pr=xpr[0], sfc=xsfc[0])
        try: os.rename(cp, cachepath)
        except: print("wtf tmp doesn't exist???", cp, date)

        return xpr[0],xsfc[0]

def date2idx(date,source,c):
    idx = c.instance_dates_dict[date][source]
    return idx

def get_sample_idxs(sample_idx,config):
    c = config
    sample_date = c.sample_dates[sample_idx]
    iidx_at_date = lambda idate,sources : [date2idx(idate,s,c) for s in sources]
    sample_idx = [iidx_at_date(sample_date+timedelta(hours=dt),sources) for dt,sources in c.sample_descriptor]
    return sample_idx

def instance2tensor(instance,is_output,c,valid_at=None):
    F = torch.HalfTensor
    mesh = c.instance_meshes[instance.idx]
    #print("instance2tensor mesh", mesh.extra_sfc_vars, mesh.extra_sfc_pad)
    pr, sfc = instance.data_pr, instance.data_sfc
    xpr = F(pr)
    xsfc = F(sfc)
    x_pr_flat = torch.flatten(xpr, start_dim=-2)
    x = torch.cat((x_pr_flat, xsfc), -1)
    

    if len(mesh.input_levels) != len(mesh.levels):
        last_lev = mesh.input_levels
        #print("uh oh", mesh.source, "input", mesh.input_levels, "output", mesh.levels)
        for inter_lev in mesh.intermediate_levels+[mesh.levels]:
            #print(len(last_lev),'->', len(inter_lev))
            #print("uh doing interp levels", last_lev, inter_lev)
            #print("befmeans", list(round(float(x), 2) for x in torch.mean(x, axis=(0,1))))
            #print("befstds", list(round(float(x), 2) for x in torch.std(x, axis=(0,1))))
            x = interp_levels(x,mesh,last_lev,inter_lev)
            #print("means", list(round(float(x), 2) for x in torch.mean(x, axis=(0,1))))
            #print("stds", list(round(float(x), 2) for x in torch.std(x, axis=(0,1))))
            last_lev = inter_lev
        #print("done")
    clamp = c.clamp_output if is_output else c.clamp
    x = torch.clamp(x, -clamp, clamp)

    if mesh.extra_sfc_pad > 0:
        before = x.shape
        zs = list(x.shape)[:-1] + [mesh.extra_sfc_pad]
        x = torch.cat((x, torch.zeros(zs)), axis=-1)
        #print("padding a bunch!", "from", before, "to", x.shape, x[..., -3:])
    assert x.shape[-1] == mesh.n_vars, f"{x.shape} vs {mesh.n_vars}, {mesh.source}"

    if valid_at is not None:
        x = set_metadata(x,valid_at)
    return x 

def get_date_tots_local(dates,mesh, only_at_z=None):
    def inner_tots(path):
        ls = os.listdir(path)
        tots = set()
        for dm in ls:
            if dm.endswith('.npz'): 
                ls2 = [dm]
            else: 
                def isyyyymm(x):
                    if len(x) != 6: return False
                    if not x.isdigit(): return False
                    return True
                if not isyyyymm(dm): continue
                try: ls2 = os.listdir(base + "/" + dm)
                except:
                    print("uh oh skipping",base, dm)
                    continue
            for x in ls2:
                if "tmp" not in x and not x.startswith("."):
                    tots.add(sdate2nix(x.split(".")[0]))
        return tots
    
    base = get_proc_path_base(mesh)
    tots = inner_tots(base)
    for extra in mesh.extra_pressure_vars + mesh.extra_sfc_vars:
        base_e = get_proc_path_base(mesh,extra=extra)
        tots = tots.intersection(inner_tots(base_e))


    tots2 = set()

    if only_at_z is None:
        l = range(0, 24, 3)
    else:
        l = only_at_z
    for d in dates:
        for h in l:
            nix = int(to_unix(d + timedelta(hours=h)))
            if nix in tots: 
                tots2.add(nix) 
    dh = np.diff(sorted(list(tots2))) // 3600
    if len(dh) == 0:
        print("weird dh is none")
        dh = np.array([0])
    
    assert len(tots2) > 0, f"no dates found in {base} for the requested dates"
    if len(tots2) == 1:
        print("Weird len for tots2")
        tots2.add(list(tots2)[0] - 1)
    print(f"{mesh.source:>8} | {base:<23} num: {len(tots2):<8} | {get_date(min(tots2)).strftime('%Y-%m-%d')} to {get_date(max(tots2)).strftime('%Y-%m-%d')} | min dh {np.min(dh):<2}, max dh {np.max(dh):<2}, median dh {np.median(dh):<4}")
    return tots2

def get_date_tots_cloud(dates,mesh, only_at_z=None):
    tots = set()
    if only_at_z is None:
        l = range(0, 24, 3)
    else:
        l = only_at_z
    for d in dates:
        for h in l:
            tots.add(to_unix(d + timedelta(hours=h)))
    return tots

class NeoWeatherDataset():
    def __init__(self,config):
        self.config = config
        self.inputs = config.inputs
        self.outputs = config.outputs
        self.clamp = config.clamp
        self.clamp_output = config.clamp_output
        self.timesteps = config.timesteps
        self.additional_vars = config.additional_vars
        self.radiation_cache = {}
        self.config.CLOUD = self.inputs[0].CLOUD
        for d in self.inputs + self.outputs:
            assert d.CLOUD == self.config.CLOUD, "All datasets must be cloud or not cloud"
        if self.config.CLOUD:
            self.config.WeatherInstanceType = WeatherInstanceCloud
            assert len(self.inputs) == 1, "Cloud only supports one input"
            assert len(self.outputs) == 1, "Cloud only supports one output"
            #assert self.inputs[0].source == "era5-28", "Cloud only supports era5 input"
            #assert self.outputs[0].source == "era5-28", "Cloud only supports era5 output"
            self.config.norms, _ = load_state_norm(self.inputs[0].wh_lev,self.inputs[0])
        else:
            self.config.WeatherInstanceType = WeatherInstanceLocal
 
    def get_sample(self,idx):
        c = self.config
        idxs = get_sample_idxs(idx,c) # list of lists of idxs
        #print("idxs", idxs, "inputs", [x.source for x in self.inputs])
        instances = mapll(lambda idx : c.WeatherInstanceType(idx,c),idxs)
        mapll(lambda inst: inst.load(),instances)
        sample = self.build_sample(instances)
        mapll(lambda inst: inst.cleanup(),instances)
        return sample
    
    def __len__(self):
        return self.config.N_samples
    
    def __getitem__(self,idx):
        return self.get_sample(idx)

    #@TIMEIT(thresh=8)
    def build_sample(self,instances):
        # the output of this function is like: [[in_tensor, in_tensor, ..., time], [out_tensor, out_tensor, ..., time], [out_tensor, out_tensor, ..., time], ...]
        c = self.config
        #print("building sample")
        #for inst in instances[0]:
        #    pprint(inst['data_pr'].flatten()[0])
        in_ = [[instance2tensor(inst,False,self.config,valid_at=to_unix(instances[0][0].date)) for inst in instances[0]]+[to_unix(instances[0][0].date)]]
        out_ = [[instance2tensor(inst,True,self.config,valid_at=to_unix(instances[i][0].date)) for inst in instances[i]]+[to_unix(instances[i][0].date)] for i in range(1,len(instances))]
        return in_ + out_

    @TIMEIT(thresh=2)
    def extend_and_clamp_input(self,x_flat,date):
        soy = date.replace(month=1, day=1)
        toy = int((date - soy).total_seconds()/86400)
        radiation = self.get_radiation(toy)[:720//self.grid.subsamp,:,np.newaxis]
        if len(x_flat.shape) > 3:
            radiation = radiation.unsqueeze(0)
        timeofday = torch.zeros_like(radiation) + date.hour/24
        dimprint("yo", radiation.shape, timeofday.shape)
        x = torch.cat((x_flat, radiation, timeofday), axis=-1)
        x = torch.clamp(x, -self.clamp, self.clamp)
        return x

    def get_radiation(self, toy):
        if toy in self.radiation_cache:
            return self.radiation_cache[toy]
        else:
            rad = (torch.HalfTensor(np.load(CONSTS_PATH+'/radiation_%d/%d.npy' % (self.grid.subsamp, toy))) - 300) / 400
            self.radiation_cache[toy] = rad
            return rad  

    @TIMEIT()
    def check_for_dates(self,dates=None):
        if self.config.use_mmap:
            os.makedirs(self.config.mmap_cache_path,exist_ok=True)
            mfs = os.listdir(self.config.mmap_cache_path)
            directory = self.config.mmap_cache_path
            size = sum(os.path.getsize(os.path.join(directory, f)) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)))
            print(f"mmap cache at {self.config.mmap_cache_path}, size: {size/2**30:0.2f} GiB, {len(mfs)} files")

        if dates is None:
            dates = self.config.requested_dates
        c = self.config

        tots_fn = get_date_tots_local if not c.CLOUD else get_date_tots_cloud 

        if c.only_at_z is not None:
            oazp = set(c.only_at_z)
            for x in c.only_at_z:
                for dh in self.timesteps:
                    oazp.add((x+dh)%24)
            oazp = sorted(list(oazp))
        else:
            oazp = None

        input_dataset_times = [tots_fn(dates,mesh, oazp) for mesh in self.inputs]
        output_dataset_times = [tots_fn(dates,mesh, oazp) for mesh in self.outputs]
        input_dataset_sources = ([mesh.source for mesh in self.inputs]) # no longer sorted
        output_dataset_sources = ([mesh.source for mesh in self.outputs]) # no longer sorted!!!!!
        assert len(input_dataset_sources) == len(set(input_dataset_sources)), "Why, why would you do this"
        assert len(output_dataset_sources) == len(set(output_dataset_sources)), "Why, why would you do this"
        c.sample_descriptor = [(0,input_dataset_sources)] + [(t,output_dataset_sources) for t in self.timesteps]
        dts = [[dt for _ in source] for dt,source in c.sample_descriptor]
        c.dt_order = flattenll(dts)
        c.instances_per_sample = len(c.dt_order)
        #all_sources = set(input_dataset_sources + output_dataset_sources)

        def add_s(set_,int_):
            return {x + int_ for x in set_}
        sample_times = set.intersection(*input_dataset_times)
        assert len(sample_times), "Input datasets don't intersect"
        #print("step0", len(sample_times), "output", len(output_dataset_times[0]))
        dd = np.diff(sorted(list(sample_times)))/3600.
        #print("uhh", np.mean(dd), np.median(dd), np.min(dd), np.max(dd))
        for dh in self.timesteps:
            sample_times = set.intersection(sample_times,*[add_s(x,-dh*3600) for x in output_dataset_times])
            #print("after", dh, "got", len(sample_times))
        #print("step1", len(sample_times))
        if c.only_at_z is not None: sample_times = set(x for x in sample_times if (datetime(1970,1,1)+timedelta(seconds=int(x))).hour in c.only_at_z)
        #print("stepd", len(sample_times))
        output_instance_times = set.union(*[add_s(sample_times,+dh*3600) for dh in self.timesteps])
        instance_times = set.union(sample_times,output_instance_times)
        active_instance_sets = [sample_times]*len(self.inputs) + [output_instance_times]*len(self.outputs)
        #output_instance
        unquie_instances = set.union(*[{(t,mesh) for t in set_} for set_,mesh in zip(active_instance_sets,self.inputs+self.outputs)])
        instances_ = sorted([(t,(t in output_instance_times and mesh.source in output_dataset_sources),mesh) for t,mesh in unquie_instances],key=lambda x: (x[0],x[1],x[2].source))
        instance_dates_dict = {}
        for i,(t,_,mesh) in enumerate(instances_):
            subd = instance_dates_dict.get(get_date(t),{})
            subd[mesh.source] = i
            instance_dates_dict[get_date(t)] = subd
        c.instance_dates_dict = instance_dates_dict
        c.sample_dates = sorted([get_date(x) for x in sample_times])
        c.instance2sample_start_dict = {date2idx(date,input_dataset_sources[0],c):i for i,date in enumerate(c.sample_dates)}
        c.instance_dates = [get_date(t) for t,_,_ in instances_]
        c.instance_meshes = [mesh for _,_,mesh in instances_]
        c.sample_dates_dict = {d:i for i,d in enumerate(c.sample_dates)}
        #c.instance_dates_dict = {d:i for i,d in enumerate(c.instance_dates)}
        c.N = len(c.instance_dates)
        if c.name == "val":
            pass
            #print("yo wtf instance dates are", c.instance_dates)
            #print("sample dates", c.sample_dates)
            #print("only at z", c.only_at_z)
        c.N_samples = len(c.sample_dates)
        #print("uh instance times is", instance_times)
        #print("len sample dates", len(c.sample_dates), c.sample_dates, "only at z", c.only_at_z)
        if len(instance_times) > 1:
            c.min_date_dh = np.min(np.diff(sorted(list(instance_times)))) // 3600
        c.min_date_dh = 1
        c.sample_range = (max(c.timesteps) // c.min_date_dh+1) * (len(self.inputs)+len(self.outputs))# the max range a sample can span
        #assert c.N  == int(max(c.output_instance_dates_dict.values()) + 1), f"{c.N} vs {max(c.output_instance_dates_dict.values()) + 1}"
        assert c.min_date_dh >= 1 or c.min_date_dh == 0, f"{c.min_date_dh} is too small until we are actually doing sub 3hr, then change this"
        assert c.sample_range > 1, f"sample range really aught to be >1, this is bad. {c.sample_range}"
        if max(c.timesteps) > 0:
            assert c.N_samples < c.N, f"Realy sus. {c.N_samples} vs {c.N}"

class EarthSpecificModel(torch.nn.Module):
    def __init__(self, mesh):
        super().__init__()
        const_vars = []
        self.n_const_vars = 0
        to_cat = []
        latlon = torch.FloatTensor(mesh.xpos)
        const_vars += ['lat','lon']; self.n_const_vars += 2; to_cat += [latlon]

        land_mask_np = np.load(CONSTS_PATH+'/land_mask.npy')
        land_mask = torch.BoolTensor(np.round(self.downsample(land_mask_np, mesh.xpos.shape)))
        const_vars += ['land_mask']; self.n_const_vars += 1; to_cat += [land_mask.unsqueeze(-1)]

        soil_type_np = np.load(CONSTS_PATH+'/soil_type.npy')
        soil_type_np = self.downsample(soil_type_np, mesh.xpos.shape,reduce=np.min)
        soil_type = torch.BoolTensor(self.to_onehot(soil_type_np))
        const_vars += ['soil_type']; self.n_const_vars += soil_type.shape[-1]; to_cat += [soil_type]

        elevation_np = np.load(CONSTS_PATH+'/topography.npy')
        elevation_np = self.downsample(elevation_np, mesh.xpos.shape,reduce=np.mean)
        elevation_np = elevation_np / np.std(elevation_np)
        elevation = torch.FloatTensor(elevation_np)
        const_vars += ['elevation']; self.n_const_vars += 1; to_cat += [elevation.unsqueeze(-1)]

        const_data = torch.cat(to_cat, axis=-1)
        self.register_buffer('const_data', const_data)

    @staticmethod
    def downsample(mask,shape,reduce=np.mean):
        dlat = (mask.shape[0]-1) // shape[0]
        dlon = mask.shape[1] // shape[1]
        assert dlon == dlat
        d = dlat
        toshape = (shape[0], d, shape[1], d)
        #fuck the south pole
        ret = reduce(mask[:-1,:].reshape(toshape),axis=(1,3)) 
        assert ret.shape == shape[:2], (ret.shape, shape[:2])
        return ret

    @staticmethod 
    def to_onehot(x):
        x = x.astype(int)
        D = np.max(x)+1
        return np.eye(D)[x]

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