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


nope = False
#nope = True
device = 'cuda'
epochs = 100
learning_rate = 1e-4
image_size = 64
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
train_loader = DataLoader(data, batch_size=1, shuffle=True)

# Model, optimizer, and loss function
model = UNet(
    in_channels=mesh.n_sfc_vars,
    out_channels=mesh.n_sfc_vars,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
    num_res_blocks=2,
    time_emb_dim=256
).to(device)

model_c = get_serp3bachelor().to(device)
model_c.eval()

total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Vars: {mesh.n_sfc_vars}, Total parameters: {total_params:0.2f}M")


optimizer = optim.Adam(model.parameters(), lr=learning_rate)



log_dir = f'{"/tmp/" if nope else ""}localruns/diffusion_{time.time():.0f}'
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)


# Training loop
for epoch in range(epochs):
    for step, (x, _) in enumerate(train_loader):
        #x_0 = x_0[0][:,100:308,900:1204,-mesh.n_sfc_vars:]
        x = [xx.to(device) for xx in x]
        xd = x[0][:,:,720:,-mesh.n_sfc_vars:].clone()
        with torch.no_grad():
            x[0][:,:,:,-3:] = 0 # need to zero this out for bachelor
            c = model_c(x,{0:'E'})[0]
            print(c.shape)
            c = c.view(*model_c.config.resolution,model_c.config.hidden_dim) # D H W C
            c = c[-1,:,3+90:-3] # -1 is for surface 
            c = c.permute(2,0,1).unsqueeze(0)
        x = xd.permute(0,3,1,2).float().to(device)
        B = x.size(0)
 

        # Sample random timesteps
        t = torch.randint(0, T, (B,), device=device).long()  # (N,)

        # Noise
        noise = torch.randn_like(x)  # (N, C, H, W)

        # Get the corresponding alphas_cumprod
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)  # (N, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        # Forward diffusion process
        x_t = sqrt_alpha_cumprod_t * x + sqrt_one_minus_alpha_cumprod_t * noise  # (N, C, H, W)

        # Predict the noise
        noise_pred = model(x_t, t, c)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}")
        # Logging
        if step % 2 == 0:
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + step)
        
        if step % 250 == 0:
            # Save model checkpoint
            checkpoint_path = os.path.join(log_dir, f'model_checkpoint_epoch_{epoch}_step_{step}.pth')
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # Optional: Log generated images
        if 0:
            with torch.no_grad():
                # Generate images from random noise
                sample = torch.randn(8, 3, image_size, image_size, device=device)
                for i in reversed(range(T)):
                    t_batch = torch.full((8,), i, device=device, dtype=torch.long)
                    noise_pred = model(sample, t_batch)

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

                # Denormalize and clip images
                generated = torch.clamp(sample, -1.0, 1.0)
                grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True, range=(-1, 1))
                writer.add_image('Generated Images', grid, epoch * len(train_loader) + step)

writer.close()
