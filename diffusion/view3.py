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

device = 'cuda'
H,W = 720,1440

tdates = get_dates([(D(2008, 1, 23), D(2009, 12, 28))])
extra = ['logtp', '15_msnswrf', '45_tcc']
mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
timesteps = [0]
data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                        timesteps=timesteps,
                                        requested_dates = tdates
                                        ))
data.check_for_dates()


diffuser = UNet(
    in_channels=mesh.n_sfc_vars,
    out_channels=mesh.n_sfc_vars,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
    num_res_blocks=2,
    time_emb_dim=256
).to(device)

forecaster = get_serp3bachelor().to(device)
forecaster.eval() 


#load_path = '/fast/djohn/localruns/diffusion_1728539115/model_checkpoint_epoch_0_step_24250.pth'
#load_path = '/fast/djohn/localruns/diffusion_1728607602/model_checkpoint_epoch_0_step_10250.pth'
#load_path = '/fast/djohn/localruns/diffusion_1728639865/model_checkpoint_epoch_0_step_9000.pth'
#load_path = '/fast/djohn/localruns/diffusion_1728639865/model_checkpoint_epoch_0_step_9000.pth'
#load_path = '/fast/djohn/localruns/diffusion_1728774610/model_checkpoint_epoch_0_step_4750.pth'
#load_path = '/fast/djohn/localruns/diffusion_1728774610/model_checkpoint_epoch_0_step_38250.pth'
load_path = '/fast/djohn/localruns/diffusion_1728639865/model_checkpoint_epoch_0_step_9000.pth'


checkpoint = torch.load(load_path, map_location=device)
diffuser.load_state_dict(checkpoint['model_state_dict'])
diffuser.eval()

from model_latlon.top import ForecastCombinedDiffusion
#from model_latlon_3d import ForecastStepDiffusion

model = ForecastCombinedDiffusion(forecaster=forecaster,diffuser=diffuser,T=1000)


print(f"Model loaded from {load_path}")
print(f"Loaded model state from epoch {checkpoint['epoch']}, step {checkpoint['step']}")


def save_imgs(sample):
    os.makedirs('diffusion/imgs', exist_ok=True)
    for i,v in enumerate(mesh.sfc_vars):
        img = sample[0,i,:,:].cpu().numpy()
        plt.imsave(f'diffusion/imgs/{v}.png', img)

with torch.no_grad():

    x = data[60][0]
    x = [x[0].to(device).unsqueeze(0), torch.Tensor([x[1]]).to(device)]
    gen = model.generate(x,0,steps=25).cpu().detach()
    
    ref = x[0][:,:,:,-mesh.n_sfc_vars:].cpu().detach()
    x[0][:,:,:,-3:] = 0
    reg = model.forecaster(x,[0])[0][:,:,:,-mesh.n_sfc_vars:].cpu().detach()
    os.makedirs('diffusion/imgs', exist_ok=True)
    for i,v in enumerate(mesh.sfc_vars):
        img = torch.vstack([ref[0,:,:,i],gen[0,:,:,i],reg[0,:,:,i]]).cpu().numpy()
        plt.imsave(f'diffusion/imgs/{v}.png', img)