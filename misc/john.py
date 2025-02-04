














from natten.functional import get_device_cc

cc = get_device_cc()
cc = get_device_cc(0) # Optionally select a specific GPU

print(f"Your device is SM{cc}.")

import natten

# Whether NATTEN was built with CUDA kernels, 
# and supports running them on this system.
print(natten.has_cuda())

# Whether NATTEN supports running float16 on
# the selected device.
print(natten.has_half())
print(natten.has_half(0)) # Optionally specify a GPU index.

# Whether NATTEN supports running bfloat16 on
# the selected device.
print(natten.has_bfloat())
print(natten.has_bfloat(0)) # Optionally specify a GPU index.

# Whether NATTEN supports running GEMM kernels
# on the selected device.
print(natten.has_gemm())
print(natten.has_gemm(0)) # Optionally specify a GPU index.

# Whether NATTEN supports running GEMM kernels
# in full precision on the selected device.
print(natten.has_fp32_gemm())
print(natten.has_fp32_gemm(0)) # Optionally specify a GPU index.

exit()

from train import *

import numpy as np
from multiprocessing.shared_memory import SharedMemory
import torch.multiprocessing as mp
import matplotlib.pyplot as plt 
from utils import *





exit()
mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh],
                                        outputs = [],
                                        timesteps=[0],
                                        requested_dates = get_dates((D(2021, 2, 11),D(2021, 2, 22))),
                                        ))
data.check_for_dates()
data[0]

exit()

# %% 
import tropycal.tracks as tracks
import matplotlib.pyplot as plt
import numpy as np

# %%
season = tracks.TrackDataset(basin='all',source='ibtracs').get_season(year=2022,basin='all')
season.plot()
plt.savefig('ignored/ohp.png')

# %% 
plt.clf()
for k,v in season.dict.items():
    if np.max(v['vmax']) > 100:
        print(v['name'],np.max(v['vmax']))
        storm = season.get_storm(k)
        storm.plot()
plt.savefig(f'ignored/ohp.png')

# %%

from evals.tc.tclib import *
season = get_storm('ian')

# %%

import tropycal.tracks as tracks    
basin = tracks.TrackDataset(basin='north_atlantic',source='hurdat',include_btk=False)
storm = basin.get_storm(('ian',2022))

storm.plot_models(datetime(2022,9,26))
plt.savefig('ignored/ohp.png')

exit()
import tropycal.tracks as tracks    

basin = tracks.TrackDataset(basin='north_atlantic',source='hurdat',include_btk=False)

for id,storm in basin.get_season(2022).dict.items():
    times = sorted(list(set([get_date_str(x.replace(hour=0, minute=0, second=0, microsecond=0)) for x in storm['time']])))
    print(id,times)



exit()
dt_dict = {6:3,24:6,72:12}
ts = get_rollout_times(dt_dict, time_horizon = 24*7)
print(ts)
exit()
x = torch.ones((1,10,20,3))

x = set_metadata(x,10)

print(type(x))
z = x * 1
print(type(z))
print(z.meta)




exit()
x = torch.ones((1,10,20,3))

x.ohp = 1

print(x.ohp)
print(x.meta)

exit()

def _rmse(e,weight=None):
    if weight is None:
        weight = torch.ones_like(e)
    weight_sum = torch.sum(weight)
    [np.where(e.shape == weight.shape[i])[0] for i in range(len(weight.shape))]
    return torch.sqrt(torch.einsum('bnml...,nm->l',torch.square(e),weight)/weight_sum)

e = torch.ones((1,10,20,3))
w = torch.ones((10,20))
_rmse(e,w)

exit()

A = [2,4]
N = 11
print(min_additions(A, N))

exit()

dsc1 = NeoDatasetConfig(WEATHERBENCH=1, levels=levels_tiny, CLOUD=0,source='hres')
dsc2 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='era5')
data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc1,
                                        output_mesh = meshes.LatLonGrid(config=dsc2),
                                        timesteps=[0],
                                        #requested_dates = get_dates((D(2022, 1, 5),D(2023, 8, 25))),
                                        requested_dates = get_dates((D(2022, 1, 5),D(2022, 1, 6))),
                                        odds_idc = 1,
                                        #use_mmap = True
                                        ))

model_conf = ForecastStepConfig(data.config.mesh, 
            output_mesh = data.config.output_mesh,
            patch_size=(4,8,8), 
            hidden_dim=768, 
            enc_swin_depth=0,
            dec_swin_depth=0, 
            proc_swin_depth=0, 
            adapter_swin_depth=8,
            timesteps=[0], 
            output_deltas = True,
            use_matepoint = False )

model = ForecastStepSwin3D(model_conf)

data.check_for_dates()
dat = default_collate([data[0]])



x = dat[0][0]
y = dat[1][0]
Nlev = 13 ; Npr = 5; Nsfc = 4 ; N = Nlev*Npr+Nsfc
x = torch.cat((torch.arange(0,Nlev*Npr).reshape(Nlev,Npr).permute(1,0).flatten(),torch.arange(Nlev*Npr,N)))
x = x.expand(1,720,1440,N).float()



x = x.half()



interp_levels(x,data.config.mesh,data.config.output_mesh)


exit()

H = 128
C = 4 
N = len(levels_medium) // C
idx = np.searchsorted(levels_medium,levels_tiny)
bin = [x // C for x in idx]
level_mappings = []
level_mapping_idxs = []
for i in range(N):
    idxs = [j for j, x in enumerate(bin) if x == i]
    m = nn.Linear(len(idxs)*mesh.n_pr_vars,H)
    level_mappings.append(m)
    level_mapping_idxs.append(idxs)

    




exit()

old_print = print

def print(*args,**kwargs):
    args = [YELLOW(str(x)) for x in args]
    old_print(*args,**kwargs,flush=True)


print("yoooo")

exit()
shm = SharedMemory(name='test', create=True, size=1000)


shm2.close()


exit()


@TIMEIT()
def expand_list(nums, n):
    return sorted({x + i for x in nums for i in range(-n, n+1)})

x = [np.random.randint(0,10000) for _ in range(10)]
y = expand_list(x, 2)

print(x,y)

exit()

@TIMEIT
def hi():
    print("hi")
    time.sleep(1)
    print("bye")

hi()

exit()


import gzip
import bz2
import lzma
import os

file_path = "/fast/proc/neo_1_28/199608/841514400.npz"

# Function to compress and report file size
def compress_and_report(file_path, compression_algorithm):
    compressed_file_path = file_path + compression_algorithm.__name__
    with open(file_path, 'rb') as original_file:
        with compression_algorithm(compressed_file_path, 'wb') as compressed_file:
            compressed_file.writelines(original_file)
    size = os.path.getsize(compressed_file_path)
    print(f"Size after {compression_algorithm.__name__} compression: {size} bytes")

# Apply gzip, bzip2, and lzma
compress_and_report(file_path, gzip.open)
compress_and_report(file_path, bz2.open)
compress_and_report(file_path, lzma.open)


exit()

with open('/fast/consts/normalization_delta_12h_28.pickle', 'rb') as f:
    p = pickle.load(f)
    print(p)    

exit()
device = torch.device("cuda:0")

conf = ForecastStepConfig(meshes.LatLonGrid(subsamp=1, levels=levels_medium),
                          patch_size=(4,8,8), 
                          conv_dim=768, 
                          depth=24,
                          lat_compress=True,
                          timesteps=[12,24])
model = ForecastStepSwin3D(conf).to(device)
input = torch.zeros(1,len(conf.mesh.lats),len(conf.mesh.lons),146).to(device)
output = model(input,dt=24)
print(output.shape)
exit()
def get_width(mesh,patch_size):
    dd = patch_size[1]
    outd = len(mesh.lons) // dd
    p1 = np.round(len(mesh.lons) / np.cos(mesh.lats * np.pi / 180) / 180).astype(int)
    combined = []
    group = []
    group_n = None
    for i,n in enumerate(p1[len(p1)//2:-1]):
        if group_n is None:
            group_n = n
        group.append(i)
        if len(group) >= group_n:
            combined.append((group_n,group))
            group = []
            group_n = None
    combined.append((group_n,group))

    print(combined)

    pass
    

#get_width(mesh,(4,8,8))

def simp(mesh):
    np.where(np.cos(np.deg2rad(mesh.lats)) < 0.75)








exit()
print(model)

for m,p in model.named_parameters():
    print(m,p.numel())

    