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
from diffusion.model import UNet, get_timestep_embedding
from model_latlon_3d import ForecastStepDiffusion

device = 'cuda'
H,W = 720,1440

tdates = get_dates([(D(2008, 1, 23), D(2009, 12, 28))])
extra = ['logtp', '15_msnswrf', '45_tcc']
mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
timesteps = [24]
data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                        timesteps=timesteps,
                                        requested_dates = tdates
                                        ))
data.check_for_dates()


# Diffusion hyperparameters
T = 1000
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T).to(device)  # (T,)

alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # (T,)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)


forecaster = get_serp3bachelor()
forecaster.eval()
del forecaster.decoders
forecaster.decoders = None
for param in forecaster.parameters():
    param.requires_grad = False


diffuser = UNet(
    in_channels=mesh.n_sfc_vars,
    out_channels=mesh.n_sfc_vars,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
    num_res_blocks=2,
    time_emb_dim=256
)

model = ForecastStepDiffusion(forecaster=forecaster,diffuser=diffuser)

load_path = '/fast/djohn/localruns/diffusion_1728539115/model_checkpoint_epoch_0_step_24250.pth'
load_path = '/fast/djohn/localruns/diffusion_1728607602/model_checkpoint_epoch_0_step_10250.pth'
load_path = '/fast/djohn/localruns/diffusion_1728639865/model_checkpoint_epoch_0_step_9000.pth'
#load_path = '/huge/deep/runs_diffusion/run_diffusion_20241011-163741/model_epoch0_iter5997_step1999_loss0.216.pt'

checkpoint = torch.load(load_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() 
model.to(device)

print(f"Model loaded from {load_path}")


def save_imgs(sample):
    os.makedirs('diffusion/imgs', exist_ok=True)
    for i,v in enumerate(mesh.sfc_vars):
        img = sample[0,i,:,:].cpu().numpy()
        plt.imsave(f'diffusion/imgs/{v}.png', img)

with torch.no_grad():

    x = data[10][0]
    x = [x[0].to(device).unsqueeze(0), torch.Tensor([x[1]]).to(device)]
    x[0][:,:,:,-3:] = 0 # need to zero this out for bachelor
    print('getting conditioning')
    c = model.predict(x,24)

    # Generate images from random noise
    B = 1
    sample = torch.randn(B, mesh.n_sfc_vars, H, W, device=device)
    for i in reversed(range(T)):
        print(i)
        t_batch = torch.full((B,), i, device=device, dtype=torch.long)
        noise_pred = model.diffuser(sample, t_batch, c)

        if i > 0:
            beta_t = betas[i]
            alpha_t = alphas[i]
            alpha_cumprod_t = alphas_cumprod[i]
            alpha_cumprod_prev = alphas_cumprod[i - 1]
            variance = beta_t * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
            noise = torch.randn_like(sample) if i > 1 else torch.zeros_like(sample)
            sample = (1 / torch.sqrt(alpha_t)) * (sample - ((beta_t / sqrt_one_minus_alphas_cumprod[i]) * noise_pred)) + torch.sqrt(variance) * noise
        else:
            sample = (1 / torch.sqrt(alphas[0])) * (sample - ((betas[0] / sqrt_one_minus_alphas_cumprod[0]) * noise_pred))

        if i % 100 == 0:
            save_imgs(sample)

    save_imgs(sample)
    generated = torch.clamp(sample, -1.0, 1.0)

