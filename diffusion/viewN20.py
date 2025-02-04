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
from model_latlon_3d import ForecastStepDiffusion

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


from diffusion.model import UNet

from model_latlon.top import ForecastModel, ForecastModelConfig, ForecastCombinedDiffusion
extra = ['logtp', '15_msnswrf', '45_tcc']
mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
conf = ForecastModelConfig(mesh,encdec_tr_depth=4,oldenc=True,olddec=True,latent_size=896,window_size=(3,5,7))

forecaster = get_serp3bachelor().to(device)
forecaster.eval() 

data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                timesteps=[6],
                                requested_dates = get_dates([(D(2009, 1, 23), D(2009, 2, 5))])
                                ))

data.check_for_dates()

diffuser = UNet(
    in_channels=mesh.n_sfc_vars,
    out_channels=mesh.n_sfc_vars,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
    num_res_blocks=2,
    time_emb_dim=256
)

load_path = '/fast/djohn/localruns/diffusion_1728639865/model_checkpoint_epoch_0_step_4000.pth'

checkpoint = torch.load(load_path, map_location=device)
diffuser.load_state_dict(checkpoint['model_state_dict'])
diffuser.eval()


model = ForecastCombinedDiffusion(forecaster=forecaster,diffuser=diffuser)

model.eval()
model.to('cuda')


print(f"Model loaded from {load_path}")


def save_imgs(sample):
    os.makedirs('diffusion/imgs', exist_ok=True)
    for i,v in enumerate(mesh.sfc_vars):
        img = sample[0,i,:,:].cpu().numpy()
        plt.imsave(f'diffusion/imgs/{v}.png', img)

with torch.no_grad():

    x = data[0][0]
    x = [x[0].to(device).unsqueeze(0), torch.Tensor([x[1]]).to(device)]
    with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
        gen = model.generate(x,0,steps=25).cpu().detach()
    
    ref = x[0][:,:,:,-mesh.n_sfc_vars:].cpu().detach()
    x[0][:,:,:,-3:] = 0
    with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
        reg = model.forecaster(x,[0])[0][:,:,:,-mesh.n_sfc_vars:].cpu().detach()
    os.makedirs('diffusion/imgs', exist_ok=True)
    for i,v in enumerate(mesh.sfc_vars):
        img = torch.vstack([ref[0,:,:,i],gen[0,:,:,i],reg[0,:,:,i]]).cpu().numpy()
        plt.imsave(f'diffusion/imgs/{v}.png', img)