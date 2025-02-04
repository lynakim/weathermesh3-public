import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import meshes
from utils import levels_medium, get_dates, levels_joank, D
from data import WeatherDataset, DataConfig
import os
from evals.package_neo import get_serp3bachelor
from diffusion.model import UNet
from neovis import Neovis
from evals.package_neo import get_brownian

device = 'cuda'

model = get_brownian()
model.eval()
model.to(device)

mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=['logtp', '15_msnswrf', '45_tcc'], input_levels=levels_medium, levels=levels_joank)

tdates = get_dates([(D(2008, 1, 23), D(2009, 12, 28))])
timesteps = [24]
data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                        timesteps=timesteps,
                                        requested_dates = tdates
                                        ))
data.check_for_dates()


s = data[0]
with torch.no_grad():
    x = s[0]
    x = [x[0].to(device).unsqueeze(0), torch.Tensor([x[1]]).to(device)]
    ref = s[1][0][:,:,-mesh.n_sfc_vars:].cpu().detach().unsqueeze(0)
    with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
        gens = model.generate(x,timesteps[0],steps=8,num=5)
        
        #x[0][:,:,:,-3:] = 0
        
        save_path= '/fast/to_srv/diffusion/Nov27-2/' 
        os.makedirs(save_path, exist_ok=True)
        for j,gen in enumerate(gens):
            for i,v in enumerate(mesh.sfc_vars):
                #img = torch.vstack([ref[0,:,:,i],gen[0,:,:,i]]).cpu().numpy()
                img = gen[0,:,:,i].cpu().numpy()
                p = f'{save_path}/var={v},num={j},_neovis.png'
                print(f"saving {p}")
                plt.imsave(p, img)
    
n = Neovis(save_path)
n.make()
    
            
