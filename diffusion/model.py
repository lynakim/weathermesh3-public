import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: (N,) tensor of time steps
    :param embedding_dim: the dimension of the embedding vector
    :return: (N, embedding_dim) tensor of embeddings
    """
    half_dim = embedding_dim // 2
    timesteps = timesteps.float()
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps[:, None] * emb[None, :]  # (N, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    return emb  # (N, embedding_dim)


def earth_padinit2d(x, pad):
    """ 
    :param x: (N, C, H, W) input tensor
    :param pad: (H_pad, W_pad) tuple of padding
    :return: (N, C, H + 2 * H_pad, W + 2 * W_pad) padded tensor
    """
    H,W = x.shape[2], x.shape[3]
    assert H / W == 0.5, f"H/W ratio must be 0.5, this is an earth grid. Got {H}/{W}. x.shape={x.shape}"
    H_pad, W_pad = pad

    new_shape = (x.shape[0], x.shape[1], H + 2 * H_pad, W + 2 * W_pad)
    new_x = x.new_zeros(new_shape)
    new_x[:,:,H_pad:H+H_pad,W_pad:W+W_pad] = x
    return new_x

def earth_padcp2d(x, pad):
    """
    Do the copy part of a the pad operation for a tensor on a earth grid.

    :param x: (N, C, H + 2 * H_pad, W + 2 * W_pad) input tensor
    :param pad: (H_pad, W_pad) tuple of padding
    :return: (N, C, H + 2 * H_pad, W + 2 * W_pad) padded tensor
    """
    H_pad, W_pad = pad
    H,W = x.shape[2] - 2*H_pad, x.shape[3] - 2*W_pad
    assert H / W == 0.5, f"H/W ratio must be 0.5, this is an earth grid. Got {H}/{W}. x.shape={x.shape}"

    if W_pad > 0: #wrap for longitude
        x[:,:,:,:W_pad] = x[:,:,:,-2*W_pad:-W_pad]
        x[:,:,:,-W_pad:] = x[:,:,:,W_pad:2*W_pad]

    if H_pad > 0: # reflect for latitude
        x[:,:,:H_pad,:] = x[:,:,2*H_pad:H_pad:-1,:]
        x[:,:,-H_pad:,:] = x[:,:,-H_pad-2:-H_pad-2*H_pad:-1,:]
    
    return x 

def earth_pad2d(x, pad):
    """ 
    :param x: (N, C, H, W) input tensor
    :param pad: (H_pad, W_pad) tuple of padding
    :return: (N, C, H + 2 * H_pad, W + 2 * W_pad) padded tensor
    """
    H,W = x.shape[2], x.shape[3]
    #assert H / W == 0.5, f"H/W ratio must be 0.5, this is an earth grid. Got {H}/{W}. x.shape={x.shape}"
    H_pad, W_pad = pad

    H_pad, W_pad = pad
    # Step 1: Pad left and right (circular padding)
    x_padded = F.pad(x, (W_pad, W_pad, 0, 0), mode='circular')
    
    # Step 2: Pad top and bottom (reflection padding)
    x_padded = F.pad(x_padded, (0, 0, H_pad, H_pad), mode='reflect')
    return x_padded


def earth_unpad2d(x, pad):
    """
    :param x: (N, C, H + 2 * H_pad, W + 2 * W_pad) input tensor
    :param pad: (H_pad, W_pad) tuple of padding
    :return: (N, C, H, W) padded tensor
    """
    H_pad, W_pad = pad
    H,W = x.shape[2] - 2*H_pad, x.shape[3] - 2*W_pad
    assert H / W == 0.5, f"H/W ratio must be 0.5, this is an earth grid. Got {H}/{W}. x.shape={x.shape}"

    return x[:,:,H_pad:H+H_pad,W_pad:W+W_pad]


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super(ResBlock, self).__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0)

        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t):
        h = self.norm1(x)
        h = F.silu(h)
        h = earth_pad2d(h, (1,1))
        h = self.conv1(h)

        time_emb = self.time_mlp(t)
        time_emb = time_emb[:, :, None, None]  # (N, C, 1, 1)
        h = h + time_emb

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = earth_pad2d(h, (1,1))
        h = self.conv2(h)

        return h + self.nin_shortcut(x)

class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        conditioning_input_channels = 896,
        conditioning_channels=256,
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels  
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        prev_channels = in_channels
        curr_channels = base_channels
        self.input_conv = nn.Conv2d(
            prev_channels, curr_channels, kernel_size=3, padding=1
        )
        prev_channels = curr_channels

        # Downsampling path
        self.skip_channels = []
        for mult in channel_mults:
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                l = 0
                if len(self.downs) in [9,10]: # conditioning
                   l = conditioning_channels
                resblock = ResBlock(prev_channels+l, out_channels, time_emb_dim)
                self.downs.append(resblock)
                prev_channels = out_channels
                self.skip_channels.append(prev_channels)  # Track skip connection channels
            downsample = nn.Conv2d(
                prev_channels,
                prev_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.downs.append(downsample)

        # Bottleneck
        self.conditioning_projection = nn.Linear(conditioning_input_channels, conditioning_channels)
        self.conditioning_conv = nn.Conv2d(conditioning_input_channels, conditioning_channels, kernel_size=2, stride=2)
        self.bottleneck = ResBlock(prev_channels + conditioning_channels, prev_channels, time_emb_dim)

        # Upsampling path
        for mult in reversed(channel_mults):
            out_channels = base_channels * mult
            upsample = nn.ConvTranspose2d(
                prev_channels,
                prev_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            self.ups.append(upsample)
            l = 0
            if len(self.ups) in [1,2]: # conditioning
                l = conditioning_channels
            for _ in range(num_res_blocks):
                resblock = ResBlock(prev_channels + self.skip_channels.pop() + l, out_channels, time_emb_dim)
                self.ups.append(resblock)
                prev_channels = out_channels

        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, prev_channels),
            nn.SiLU(),
            nn.Conv2d(prev_channels, self.out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t, c):
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        x = self.input_conv(x)
        hs = []
        B = x.shape[0]
        cp = self.conditioning_projection(c.permute(0,2,3,1)).permute(0,3,1,2)

        c = c.repeat(B,1,1,1)
        cp = cp.repeat(B,1,1,1)

        # Downsampling path
        for i,layer in enumerate(self.downs):
            #print(i,int(isinstance(layer, ResBlock)),x.shape)
            if i in [9,10]: # conditioning
                assert isinstance(layer, ResBlock)
                x = torch.cat([x, cp], dim=1)
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
                hs.append(x)  # Store skip connection
            else:  # Downsample
                x = layer(x)

        #print(x.shape)
        # Bottleneck
        cc = self.conditioning_conv(c)
        x = torch.cat([x, cc], dim=1)
        x = self.bottleneck(x, t_emb)

        # Upsampling path
        for i,layer in enumerate(self.ups):
            #print(i,int(isinstance(layer, ResBlock)),x.shape)
            if i in [1,2]:
                assert isinstance(layer, ResBlock)
                x = torch.cat([x, cp], dim=1)
            if isinstance(layer, nn.ConvTranspose2d):  # Upsample
                x = layer(x)
            else:  # ResBlock
                skip = hs.pop()
                x = torch.cat([x, skip], dim=1)  # Concatenate skip connection
                x = layer(x, t_emb)

        x = self.output_conv(x)
        return x






if __name__ == '__main__':
# Model, optimizer, and loss function
    model = UNet(
        in_channels=7,
        out_channels=7,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256
    )
    print(model)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total_params:0.2f}M")

    # Generate random input
    B, C, H, W = 1, 7, 720, 720
    x = torch.randn(B, C, H, W)
    c = torch.randn(B, 896, H//8 , W//8)
    t = torch.randint(0, 1000, (B,)).long()
    model.eval()
    with torch.no_grad():
        y = model(x, t, c) 
