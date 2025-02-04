import sys
import os
sys.path.append('/fast/wbjoan10/deep')
from utils import *
from eval import unnorm_output, unnorm, unnorm_output_partial, compute_errors, all_metric_fns
import pickle
from dataloader import NeoLoader, NeoDataConfig
from data import *
from model_latlon_3d import *
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader    
import json
import hashlib
import base64
from evals import package_neo

model = package_neo.get_shortking()
print("mp", model.config.use_matepoint)
a = int(sys.argv[1])
Aa = a
b = int(sys.argv[2])


DH = 24
DH = 48

import random
random.seed(DH)
torch.manual_seed(DH)

dd = timedelta(hours=24)
dates = get_dates([(D(1990, 1, 23), D(2019, 12, 28), dd), (D(2021, 1, 1), D(2022, 7, 1), dd)])

dates = np.array_split(dates, b)[a]

inputs = model.config.inputs 
outputs = model.config.outputs

data = NeoWeatherDataset(NeoDataConfig(inputs=inputs, outputs=outputs,
                        timesteps=[DH],
                        requested_dates = dates,
                        use_mmap = False,
                        only_at_z = [0,6,12,18],
                        clamp_output = np.inf,
                        ))

data.check_for_dates()

dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=2, collate_fn=default_collate)

model = model.to('cuda')

with torch.no_grad():
    model = model.half()
    mesh = data.config.outputs[0]
    #for i in range(len(data)):
    for sample in dataloader:
        AAA = []
        neonix = int(sample[-1][-1])
        dd = datetime(1970,1,1)+timedelta(seconds=neonix)
        bb = "/fast/proc/shortking/f%03d/%04d%02d"%(DH, dd.year, dd.month)
        if os.path.exists(bb+"/%d.npz"%(neonix)): continue
        #dt = data.config.sample_descriptor[-1][0] #dt = sample_dts(sample)[0]
        #sample = default_collate([data[i]])
        x = sample[0]
        #print("uhhhh sample", len(sample), [len(a) for a in sample], sample[0][-1], sample[1][-1])
        x = [xx.to('cuda') for xx in x]
        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            #print("hey uh", model, x)
            ys = model.rollout(x,time_horizon=DH,min_dt=DH) 
        #for i, y in enumerate(ys.items()):
        #    print("hey uh", i, y) 
        for j,(ddt,y) in enumerate(sorted(ys.items())):
            #nix = to_unix(data.config.sample_dates[i])
            #print("heyooo", ddt, nix, x[-1], sample[j+1][-1], len(ys))
            #assert sample[j+1][-1] == nix+ddt*3600, f"{nix} {ddt} {sample[j+1][-1]} {j} sample date mismatch"
            #y = y.to('cuda')
            xu = unnorm(x[0],data.inputs[0])
            assert y.meta.delta_info is None
            #y = unnorm(y,data.outputs[0])
            #neonix = int(nix+ddt*3600)
            neonix = int(sample[j+1][-1])
            dd = datetime(1970,1,1)+timedelta(seconds=neonix)
            bb = "/fast/proc/shortking/f%03d/%04d%02d"%(DH, dd.year, dd.month)
            yexp = y[..., :144].numpy()
            ypr = yexp[..., :140]
            ypr.shape = (720, 1440, 5, 28)
            ysfc = yexp[..., 140:]
            ysfc.shape = (720, 1440, 4)
            os.makedirs(bb, exist_ok=True)
            np.savez(bb+"/%d.npz"%(neonix), pr=ypr, sfc=ysfc)
            print(HOSTNAME, Aa, "did", dd)
            #xu,y = unnorm_output(x[0],y,model,dt,y_is_deltas=False)
            #yt = unnorm(sample[j+1][0].to('cuda'),mesh)
            #err = (y-yt)[:,:,:28].cpu().numpy()
            #print("rms", np.sqrt(np.mean(np.square(err))))
