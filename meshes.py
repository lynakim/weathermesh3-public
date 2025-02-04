import icosphere
import pickle
import numpy as np
import torch
import uuid
import sys
from scipy.spatial import KDTree
import os
from utils import *
from utils import SourceCodeLogger
import json
import hashlib
import base64
from utils import * 
#n = int(sys.argv[1])

def haversine(lon1, lat1, lon2, lat2):
    from math import sin, cos, sqrt, atan2, radians
    R = 6371  # Radius of Earth in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2) * sin(dlat/2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2) * sin(dlon/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance


class Mesh():
    def __init__(self):
        global CONSTS_PATH,PROC_PATH,RUNS_PATH
        self.PROC_PATH = PROC_PATH ; self.CONSTS_PATH = CONSTS_PATH ; self.RUNS_PATH = RUNS_PATH
        
        self.is_required = True
        self.all_var_files = ['base']
        self.ens_num = None
        self.subsamp = 1
        self.resolution = 0.25
        
    # Post init function to be called after the child's __init__ function 
    # (kinda whack formatting but was running into a chicken and egg with the init functions so this may be the cleanest way?)
    # (if you don't like, you never saw me here....)
    def __post__init__(self):
        # Set the string id for the mesh
        # Calls the child's override first unless it is not specified
        self.set_string_id() 

    # The shape of the expected tensor in torch.Size([...]) format
    def shape(self):
        return -1 # Case where there is no shape 
    
    # Defines a unique string id to identify the mesh in the dataloader
    def set_string_id(self):
        raise NotImplementedError("set_string_id not implemented for this Mesh. You are required to define a unique string id for your mesh")

class TCRegionalIntensities(Mesh, SourceCodeLogger):
    def __init__(self):
        super().__init__()
        self.source = 'intensities-0'
        self.load_locations = ['/fast/proc/cyclones']
        super().__post__init__()
        
    def summary_str(self):
        return f"source: {self.load_locations[0]}"

    def set_string_id(self):
        json_str = self.to_json()
        hash = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
        self.string_id = hash[:8]
    
    def to_json(self):
        out = {}
        out['mesh_type'] = self.__class__.__name__
        # Set param variables to be saved in the json (and hence for string_id creation here)
        return json.dumps(out, indent=2)

class DAMesh(Mesh, SourceCodeLogger):
    def __init__(self):
        super().__init__()
        self.da_window_size = 6
        self.encoder_subset = np.inf

class SurfaceData(DAMesh):
    def __init__(self, **kwargs):
        super().__init__()
        self.disk_varlist = ["is_ship", "reltime_hours", "lat_deg", "lon_deg", "unnorm_elev_m", "unnorm_mslp_hpa", "unnorm_2t_c", "unnorm_2d_c", "unnorm_10u_mps", "unnorm_10v_mps", "unnorm_viz_m", "unnorm_3hpr_mm", "unnorm_24hpr_mm"]
        self.vars = ["is_ship", "elev", "mslp", "2t", "2d", "10u", "10v"]
        self.var_indicators = [v+"_present?" for v in self.vars]
        self.indices = ["reltime_hours","lat_deg","lon_deg"]
        self.addl = ["sin(lat_deg)","cos(lat_deg)","sin(lon_deg)","cos(lon_deg)","sin(time_of_year)","cos(time_of_year)","sin(time_of_day)","cos(time_of_day)","solar_elevation"]
        
        self.full_varlist = self.indices + self.vars + self.var_indicators + self.addl

        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a {self.__class__.__name__} attribute"
            setattr(self,k,v)
            
        super().__post__init__()

    def shape(self):
        return [-1,len(self.full_varlist)]
    
    def set_string_id(self):
        self.string_id = "sfc"

class BalloonData(DAMesh):
    def __init__(self, **kwargs):
        super().__init__()
        self.disk_varlist = ["id","reltime_seconds","pres_pa","lat_deg","lon_deg","unnorm_gp","unnorm_temp_k","unnorm_rh","unnorm_ucomp_mps","unnorm_vcomp_mps"]
        self.vars = ["gp","temp","rh","ucomp","vcomp"]
        self.var_indicators = [v+"_present?" for v in self.vars]
        self.indices = ["reltime_hours","pres_hpa","lat_deg","lon_deg"]
        self.addl = ["sin(lat_deg)","cos(lat_deg)","sin(lon_deg)","cos(lon_deg)","sin(time_of_year)","cos(time_of_year)","sin(time_of_day)","cos(time_of_day)","solar_elevation"]
        
        self.full_varlist = self.indices + self.vars + self.var_indicators + self.addl
        self.encoder_input_dtype = torch.float32
        self.encoder_subset = 2**16-2

        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a {self.__class__.__name__} attribute"
            setattr(self,k,v)
            
        super().__post__init__()

    def shape(self):
        return [-1,len(self.full_varlist)]
    
    def set_string_id(self):
        self.string_id = "igra"

# 🎉🎉🎉🎉 LET'S GOOOOOOOOOOOOOO 🎉🎉🎉🎉
class WindborneData(DAMesh): 
# 🎉🎉🎉🎉 LET'S GOOOOOOOOOOOOOO 🎉🎉🎉🎉
    def __init__(self, **kwargs):
        super().__init__()
        self.disk_varlist = ['id', 'reltime_hours', 'lat_deg', 'lon_deg', 'pres_pa', 'unnorm_ucomp_mps', 'unnorm_vcomp_mps', 'unnorm_temp_k', 'unnorm_rh', 'unnorm_sh_mgpkg']
        self.vars = ["ucomp", "vcomp", "temp", "rh", "sh"]
        self.var_indicators = [v+"_present?" for v in self.vars]
        self.indices = ["reltime_hours", "pres_hpa", "lat_deg", "lon_deg"]
        self.addl = ["sin(lat_deg)", "cos(lat_deg)", "sin(lon_deg)", "cos(lon_deg)", "sin(time_of_year)","cos(time_of_year)","sin(time_of_day)","cos(time_of_day)","solar_elevation"]
        self.full_varlist = self.indices + self.vars + self.var_indicators + self.addl
        #self.encoder_subset = 100_000
        super().__post__init__()
        
        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a {self.__class__.__name__} attribute"
            setattr(self,k,v)
            
    def set_string_id(self):
        self.string_id = "windborne"
        
    def shape(self):
        return [-1, len(self.full_varlist)]

class RadiosondeData(DAMesh):
    def __init__(self, **kwargs):
        super().__init__()
        self.disk_varlist = ['id', 'reltime_hours', 'lat_deg', 'lon_deg', 'pres_pa', 'unnorm_ucomp_mps', 'unnorm_vcomp_mps', 'unnorm_temp_k', 'unnorm_rh']
        self.vars = ["ucomp", "vcomp", "temp", "rh"]
        self.var_indicators = [v+"_present?" for v in self.vars]
        self.indices = ["reltime_hours", "pres_hpa", "lat_deg", "lon_deg"]
        self.addl = ["sin(lat_deg)", "cos(lat_deg)", "sin(lon_deg)", "cos(lon_deg)", "sin(time_of_year)","cos(time_of_year)","sin(time_of_day)","cos(time_of_day)","solar_elevation"]
        self.full_varlist = self.indices + self.vars + self.var_indicators + self.addl
        self.encoder_subset = 100_000
        super().__post__init__()
        
        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a {self.__class__.__name__} attribute"
            setattr(self,k,v)
            
    def set_string_id(self):
        self.string_id = "radiosonde"
        
    def shape(self):
        return [-1, len(self.full_varlist)]

class SatwindData(DAMesh):
    def __init__(self, **kwargs):
        super().__init__()
        self.disk_varlist = ['id', 'reltime_hours', 'lat_deg', 'lon_deg', 'satzenith_deg', 'pres_pa', 'unnorm_ucomp_mps', 'unnorm_vcomp_mps']
        self.vars = ["ucomp", "vcomp"]
        self.var_indicators = [v+"_present?" for v in self.vars]
        self.indices = ["reltime_hours", "pres_hpa", "lat_deg", "lon_deg"]
        self.addl = ["sin(lat_deg)", "cos(lat_deg)", "sin(lon_deg)", "cos(lon_deg)", "sin(satzenith_deg)", "cos(satzenith_deg)"]
        self.full_varlist = self.indices + self.vars + self.var_indicators + self.addl
        self.encoder_subset = 100_000
        super().__post__init__()
        
        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a {self.__class__.__name__} attribute"
            setattr(self,k,v)
            
    def set_string_id(self):
        self.string_id = "satwnd"
        
    def shape(self):
        return [-1, len(self.full_varlist)]

class MicrowaveData(DAMesh):
    def __init__(self, instrument, **kwargs):
        super().__init__()
        self.instrument = instrument
        self.nchan = {"atms": 22, "1bamua": 15}[instrument]
        self.sats = self.get_sat_mapping(instrument)
        self.vars = ["chan_"+str(i) for i in range(self.nchan)]
        self.var_indicators = [v+"_present?" for v in self.vars]
        self.disk_varlist = ["reltime_hours", "said", "lat_rad", "lon_rad", "saza_rad", "soza_rad"] + ["chan_"+str(i) for i in range(self.nchan)]

        """
                # x looks like this
        # - dt (always negative, -1 to 0 [hours])
        # - satellite id / said. should just convert to one hot of size len(mesh.sats) / do an embedding
        # - lat, lon: should do whatever embedding we do for point obs on the latent space presumably
        # - saza/soza: some relevant angles. should convert to sine and cosine and treat it as part of the observation
        # - self.nchan channels. occasionally some nans that should be converted to 0 and masked off with the _present? variables
        """
        self.indices = ["reltime_hours", "lat_deg", "lon_deg"]
        self.addl = ["sin(lat_rad)", "cos(lat_rad)", "sin(lon_rad)", "cos(lon_rad)", "sin(saza_rad)", "cos(saza_rad)", "sin(soza_rad)", "cos(soza_rad)"] + ["onehot_said_"+str(i) for i in range(len(self.sats))]
        self.full_varlist = self.indices + self.vars + self.var_indicators + self.addl
        self.encoder_input_dtype = torch.float16
        self.encoder_subset = 200_000
        super().__post__init__()

        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a {self.__class__.__name__} attribute"
            setattr(self,k,v)

    def set_string_id(self):
        self.string_id = self.instrument

    def get_sat_mapping(self, instrument):
        if instrument == "atms":
            return [224, 225, 226]
        elif instrument == "1bamua":
            return [3, 4, 5, 206, 207, 209, 223, 784]
        else:
            assert KeyError("unknown instrument")

    def shape(self):
        return [-1, len(self.full_varlist)] # lat lon hack, gets stripped away
    

class StationData(Mesh, SourceCodeLogger):
    def __init__(self):
        super().__init__()
        self.vars = ["tmpf", "dwpf", "mslp", '10u', '10v', 'skyl1', 'vsby', 'logprecip']
        self.metar_to_ncar = {
            "tmpf": "167_2t",
            "dwpf": "168_2d",
            "mslp": "151_msl",
            "10u": "165_10u",
            "10v": "166_10v",
            "vsby": "vsby",
            "skyl1": "skyl1",
            "logprecip": "logtp",
        }
        self.full_varlist = self.vars
        self.madis_latlons = pickle.load(open("/fast/ignored/hres/station_latlon.pickle", "rb"))
        self.n_madis = len(self.madis_latlons)
        self.metar_latlons = pickle.load(open("/fast/ignored/hres/station_latlon_old.pickle", "rb"))
        self.n_metar = len(self.metar_latlons)
        self.is_slop = pickle.load(open("/fast/ignored/hres/is_slop.pickle", "rb"))
        assert type(self.is_slop[0]) == bool
        self.is_slop = np.concatenate([self.is_slop, np.zeros(self.n_metar, dtype=bool)])
        self.all_latlons = np.concatenate([self.madis_latlons, self.metar_latlons])
        self.COLUMNS = [
            "timestamp",
            "station",
            "mslp",
            "tmpf",
            "dwpf",
            "10u",
            "10v",
            "skyl1",
            "vsby",
            "logprecip"
        ]
        super().__post__init__()
    
    def shape(self):
        return [len(self.full_varlist)]

    def set_string_id(self):
        self.string_id = "sfc_stations" # should this involve more info e.g. old new ratio?
    
    def summary_str(self):
        return f"madis + GHCNh stations: {self.n_madis}, metar + ecmwf stations: {self.n_metar}"

class LatLonGrid(Mesh, SourceCodeLogger):
    def __init__(self, **kwargs):
        super().__init__()
        self.source = 'era5-28'
        self.load_locations = ['/fast/proc/']
        self.hour_offset = 0
        
        self.input_levels = None
        self.levels = None
        self.extra_sfc_vars = []
        self.extra_sfc_pad = 0
        self.extra_pressure_vars = []
        self.intermediate_levels = []
        
        lats = np.arange(90, -90.01, -self.resolution)[:-1]
        lats.shape = (lats.shape[0]//self.subsamp, self.subsamp)
        self.lats = np.mean(lats, axis=1)

        lons = np.arange(0, 359.99, self.resolution)
        lons.shape = (lons.shape[0]//self.subsamp, self.subsamp)
        lons[lons >= 180] -= 360
        self.lons = np.mean(lons, axis=1)
        self.parent = None
        self.bbox = None
        self.precip_buckets = None
        
        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a NeoDatasetConfig attribute"
            setattr(self,k,v)
        
        self.update()
        self.update_mesh()
        if not hasattr(self, 'string_id'): super().__post__init__()

    def shape(self):
        return torch.Size([len(self.lats), len(self.lons), self.n_vars])

    def set_string_id(self):
        j, hash = self.to_json()
        dtstr = '' if self.hour_offset == 0 else f"(-{self.hour_offset}h)"
        s = f"{dtstr}{self.source}>{self.n_levels}p{self.n_pr_vars}s{self.n_sfc_vars-self.extra_sfc_pad}z{self.extra_sfc_pad}r{self.res}-{hash}"
        print(ORANGE(f"Set string_id for LatLonGrid to {s}"))
        self.string_id = s
        
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
        if self.source == "hres9km-25":
            assert len(self.levels) != 25, "to avoid bugs caused by levels_ncarhres and levels_joank both being 25 levels but different, please use a different number of levels for hres9km. Recommended: levels_ecm1"
        self.pressure_vars = core_pressure_vars + self.extra_pressure_vars
        self.core_sfc_vars = core_sfc_vars
        self.sfc_vars = core_sfc_vars + self.extra_sfc_vars + ['zeropad']*self.extra_sfc_pad
        self.varlist = self.pressure_vars + self.sfc_vars
        self.full_varlist = []
        for v in self.pressure_vars:
            self.full_varlist = self.full_varlist + [v+"_"+str(l) for l in self.levels]

        self.all_var_files = ['base'] + self.extra_pressure_vars + [v for v in self.extra_sfc_vars if 'bucket' not in v]
        self.full_varlist = self.full_varlist + self.sfc_vars

        self.n_levels = len(self.levels)
        self.n_pr_vars = len(self.pressure_vars)
        self.wh_lev = [levels_full.index(x) for x in self.levels] #which_levels
        self.n_pr_vars = len(self.pressure_vars)
        self.n_pr = self.n_levels * self.n_pr_vars
        self.n_sfc = len(self.sfc_vars)
        self.n_sfc_vars = self.n_sfc
        self.n_vars = self.n_pr + self.n_sfc

        if self.precip_buckets is not None:
            bucket_idx = [i for i, x in enumerate(self.full_varlist) if "bucket" in x]
            assert len(self.full_varlist) - len(bucket_idx) == bucket_idx[0], "buckets must be at the end of the list"
            self.num_rms_vars = len(self.full_varlist) - len(bucket_idx)
            self.n_bucket_vars = len(bucket_idx)
        else:
            assert all("bucket" not in x for x in self.sfc_vars), "you doing buckets or na?"
            self.num_rms_vars = len(self.full_varlist)
            self.n_bucket_vars = 0

        self.bucket_vars = [(i, x) for i, x in enumerate(self.full_varlist) if "bucket0" in x]

    def get_zeros(self):
        return np.zeros((1, len(self.lats), len(self.lons), self.n_vars), dtype=np.float32)
    
    def summary_str(self):
        base_pr = f"0:{self.n_pr} base pr"; cur = self.n_pr
        base_sfc = f"{cur}:{cur + len(core_sfc_vars)} base sfc"; cur += len(core_sfc_vars)
        extra_sfc = f"{cur}:{cur + len(self.extra_sfc_vars)} extra sfc"; cur += len(self.extra_sfc_vars)
        zero_pad = f"{cur}:{cur + self.extra_sfc_pad} zero pad"
        if self.extra_sfc_pad == 0:
            zero_pad = "no zero pad"

        return f"source: {self.source}, {self.n_levels} lev [ {base_pr} | {base_sfc} | {extra_sfc} | {zero_pad} ]"

    def update_mesh(self):
        self.Lons, self.Lats = np.meshgrid(self.lons, self.lats)
        self.res = self.resolution * self.subsamp
        self.Lons /= 180
        self.Lats /= 90
        self.xpos = np.stack((self.Lats, self.Lons), axis=2)

        import utils
        self.state_norm,self.state_norm_stds,self.state_norm_means = utils.load_state_norm(self.wh_lev,self,with_means=True)

    def lon2i(self, lons):
        return np.argmin(np.abs(self.lons[:,np.newaxis] - lons),axis=0)
    
    def lat2i(self, lats):
        return np.argmin(np.abs(self.lats[:,np.newaxis] - lats),axis=0)
    
    def to_json(self, model_name=None):
        out = {}
        out['mesh_type'] = self.__class__.__name__
        out['pressure_vars'] = self.pressure_vars
        out['sfc_vars'] = self.sfc_vars
        out['full_varlist'] = self.full_varlist
        out['levels'] = self.levels
        out['lats'] = self.lats.tolist()
        out['lons'] = self.lons.tolist()
        out['res'] = self.res
        if model_name is not None: out['model_name'] = model_name
        st = json.dumps(out, indent=2)
        hash_24bit = hashlib.sha256(st.encode()).digest()[:3]
        base64_encoded = base64.b64encode(hash_24bit).decode()
        # replace + and / with a and b to because / in filenames are RIP. we could use urlsafe_b64encode but it would be incompatible with the old hashes
        base64_encoded = base64_encoded.replace('+', 'a').replace('/', 'b')
        return st, base64_encoded

    @staticmethod
    def from_json(js):
        out = LatLonGrid()
        out.__dict__ = js
        out.source = "unknown"
        out.lats = np.array(out.lats)
        out.lons = np.array(out.lons)
        out.wh_lev = [levels_full.index(x) for x in out.levels] #which_levels
        out.subsamp = 1; assert np.diff(out.lats)[1] == -0.25, f'lat diff is {np.diff(out.lats)}'
        out.update_mesh()
        return out

def get_mesh(n, k, levels=None):
    import utils
    if levels is None:
        levels = utils.levels_full


    fn = f"{utils.CONSTS_PATH}/grids/grid_%d_%d_%d.pickle" % (n, k, len(levels))
    if os.path.exists(fn) and 0:
        with open(fn, "rb") as f:
            return pickle.load(f)

    g = Mesh(n, k, levels)

    g.wh_lev = [utils.levels_full.index(x) for x in levels]

    os.makedirs(f"{utils.CONSTS_PATH}/grids", exist_ok=True)
    #with open(fn, "wb") as f:
    #    pickle.dump(g, f)
    return g


def get_Dec23_meshes():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)

    return imesh, omesh
