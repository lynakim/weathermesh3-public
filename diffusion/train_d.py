import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from diffusion.model import UNet, get_timestep_embedding
from utils import * 
from data import WeatherDataset, DataConfig
import os
from evals.package_neo import get_serp3bachelor

nope = 0
#nope = True
device = torch.device('cuda:1')
name = "h128_full_earth"
epochs = 100
learning_rate = 1e-4
B = 1 # Batch size
T = 1000  # Number of diffusion steps
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T).to(device)  # (T,)

alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # (T,)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)


tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
extra = ['logtp', '15_msnswrf', '45_tcc']
mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
timesteps = [0]
data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                        timesteps=timesteps,
                                        requested_dates = tdates
                                        ))
data.check_for_dates()
train_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=2)

# Model, optimizer, and loss function
diffuser = UNet(
    in_channels=mesh.n_sfc_vars,
    out_channels=mesh.n_sfc_vars,
    base_channels=128,
    channel_mults=(1, 2, 4, 8),
    num_res_blocks=2,
    time_emb_dim=256
)#.to(device)

forecaster = get_serp3bachelor()#.to(device)
forecaster.eval() 

from model_latlon.top import ForecastCombinedDiffusion, get_bbox
model = ForecastCombinedDiffusion(forecaster=forecaster,diffuser=diffuser,T=1000)
model.to(device)


total_params = sum(p.numel() for p in model.diffuser.parameters()) / 1e6
print(f"Vars: {mesh.n_sfc_vars}, Total parameters: {total_params:0.2f}M")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

log_dir = f'{"/tmp/" if nope else ""}/huge/deep/runs_diffusion2/{name}_{time.time():.0f}'
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)


# Training loop

HALF = True

for epoch in range(epochs):
    step = 0
    for (x, _) in train_loader:

        #x_0 = x_0[0][:,100:308,900:1204,-mesh.n_sfc_vars:]
        x = [xx.to(device) for xx in x]
 
        y = x[0][:,:,:,-model.config.inputs[0].n_sfc_vars:].clone()
        C = y.shape[3] 
        for i in range(C):
            assert not (y[...,i] == 0).all(), "All zeros in y" 
        y = y.permute(0,3,1,2)
        if not HALF:
            y = y.float()
        c = None
        for i in range(10):
            t = torch.randint(0, T, (B,), device=device).long()  # (N,)

            noise = torch.randn_like(y) 
            sqrt_alpha_cumprod_t = model.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)  # (N, 1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

            # Forward diffusion process
            y_t = sqrt_alpha_cumprod_t * y + sqrt_one_minus_alpha_cumprod_t * noise  # (N, C, H, W)

            for bbox in [[0,1/3,0,1],[1/3,2/3,0,1],[2/3,1,0,1]]:
                with torch.autocast(enabled=HALF, device_type='cuda', dtype=torch.float16):
                    noise_pred,c = model(x, 0, y_t, t,c=c,bbox=bbox)

                    # Compute loss
                noise_sub = get_bbox(bbox, noise)
                loss = F.mse_loss(noise_pred, noise_sub)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1   
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}, Vram: {torch.cuda.max_memory_allocated(device=device)/1024**3:.1f}GiB")
                # Logging
                if step % 2 == 0:
                    writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + step)
                
                if step % 250 == 0:
                    # Save model checkpoint
                    checkpoint_path = os.path.join(log_dir, f'model_checkpoint_epoch_{epoch}_step_{step}.pth')
                    torch.save({
                        'epoch': epoch,
                        'step': step,
                        'model_state_dict': model.diffuser.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")

                # Optional: Log generated images
                if step % 100 == 0:
                    with torch.no_grad():                
                        model.eval()
                        gen = model.generate(x, 0,steps=25).cpu().detach()
                        save_dir = log_dir + "/diffusion_img/"
                        os.makedirs(save_dir, exist_ok=True)
                        for i,v in enumerate(mesh.sfc_vars):
                            img = gen[0,:,:,i]
                            plt.imsave(f"{save_dir}/{step}_{v}.png", img.numpy())

                        model.train()

writer.close()
