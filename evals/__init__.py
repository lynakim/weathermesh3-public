from utils import *
import os
from meshes import LatLonGrid


EVALUATION_PATH = '/huge/deep/evaluation/'

def get_errors_path(model, resolution=None): 
    deg_str = '' if resolution is None else f'_{resolution}deg'
    return f'{EVALUATION_PATH}/{model}/errors{deg_str}/'
def get_model_path(model): return f'{EVALUATION_PATH}/{model}/'


# def save_instance(x,path,mesh,downsample_levels=False):
#     if isinstance(x,torch.Tensor):
#         x = x.detach().cpu().numpy()
#     if downsample_levels:
#         newconf = NeoDatasetConfig(conf_to_copy=mesh.config,levels=levels_ecm2)
#         newmesh = type(mesh)(newconf)
#         wh_levnew = [mesh.config.levels.index(x) for x in levels_ecm2]
#         xshape_new = list(x.shape[:-1]) + [newmesh.n_vars]
#         xnew = np.zeros(xshape_new,dtype=x.dtype)
#         for i,j in enumerate(wh_levnew):
#             xnew[...,i*mesh.n_pr_vars:(i+1)*mesh.n_pr_vars] = x[...,j*mesh.n_pr_vars:(j+1)*mesh.n_pr_vars]
#         xnew[...,-mesh.n_sfc:] = x[...,-mesh.n_sfc:]
#         x = xnew
#         mesh = newmesh
#     js,hash = mesh.to_json()
#     os.makedirs(os.path.dirname(path),exist_ok=True)
#     metapath = os.path.dirname(path)+f'/meta.{hash}.json'
#     if os.path.exists(metapath):
#         with open(metapath,'r') as f:
#             js2 = f.read()
#         assert js == js2, "metadata mismatch"
#     else:    
#         with open(metapath,'w') as f:
#             f.write(js)
#     if isinstance(x,torch.Tensor):
#         x = x.detach().cpu().numpy()
#     if x.shape[0] == 1:
#         x = x[0]
#     else:
#         assert len(x.shape) == 3, "Can not be multi batch"
#     filepath= path+f".{hash}.npy"
#     print("Saving to", filepath)
#     np.save(filepath,x)
    
def load_instance(path,mmap=True,bbox=None):
    hash = os.path.basename(path).split('.')[-2]
    metapath = os.path.dirname(path)+f'/meta.{hash}.json'
    if not os.path.exists(metapath):
        print(f"Warning: metadata not found at {metapath}, defaulting to Qfiz")
        metapath = f"/huge/deep/realtime/outputs/WeatherMesh/meta.Qfiz.json"
    with open(metapath,'r') as f:
        js = json.load(f)
    mesh = LatLonGrid.from_json(js)
    x = np.load(path,mmap_mode='r' if mmap else None)
    if bbox is not None:
        newmesh = bbox_mesh(mesh,bbox)
        x = select_bbox(x,mesh,bbox)
        mesh = newmesh
    return x, mesh 


def mmap_instance(path):
    hash = os.path.basename(path).split('.')[-2]
    metapath = os.path.dirname(path)+f'/meta.{hash}.json'
    with open(metapath,'r') as f:
        js = json.load(f)
    mesh = LatLonGrid.from_json(js)
    return np.load(path,mmap_mode='r'), mesh 

from data import DataConfig, WeatherDataset
from eval import unnorm
era5_dataset = None ; era5_bbox = None
def load_era5_instance(date, bbox=None):
    global era5_dataset, era5_bbox
    if era5_dataset is None:
        era5_bbox = bbox
        mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
        if era5_bbox is not None:
            mesh = bbox_mesh(mesh,era5_bbox)
        era5_dataset = WeatherDataset(DataConfig(inputs=[mesh],
                                            outputs = [],
                                            timesteps=[0],
                                            requested_dates = get_dates((D(2020, 2, 11),D(2024, 2, 22))),
                                            ))
        era5_dataset.check_for_dates()
    assert era5_bbox == bbox, "you can only call this function with one bbox per script oops"
    mesh = era5_dataset.config.inputs[0]
    date = get_date(date)
    assert date in era5_dataset.config.instance_dates
    assert False, "This is John and I'm sorry but while deleting the neodataloader on Oct 4 I'm pretty sure I broke whatever this was doing"
    idx = date2idx(date,mesh.source,era5_dataset.config)
    x = era5_dataset[idx][0]
    assert to_unix(date) == int(x[1])
    x = unnorm(x[0],mesh)
    return x, mesh
    
    
# def to_filename(nix,dt,tags=[], always_plus=False):
#     dts = ''
#     if dt != 0 or always_plus:
#         dts = f'+{dt}'
#     tagstr = ''
#     if len(tags) > 0:
#         tagstr = '.'+'.'.join(tags)
#     return f'{get_date_str(nix)}{dts}{tagstr}'

def get_compare_tag(data_config):
    if data_config.output_mesh is None:
        return data_config.mesh.config.source
    else:
        return data_config.mesh.config.source +'->'+data_config.output_mesh.config.source 