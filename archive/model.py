import torch
import torch.utils.checkpoint
import numpy as np
import time
from torch import nn
import gc
from utils import *
from meshes import *

from einops import rearrange
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        # x is shape (N, k, D)
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        #print(q.shape, k.shape, v.shape, "x", x.shape)

        dots = torch.matmul(q * self.scale, k.transpose(-1, -2))

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            attn_x = attn(x)
            #attn_x = torch.utils.checkpoint.checkpoint(attn, x)
            x = attn_x + x
            x = ff(x) + x
        return self.norm(x[:,0,:])

class ForecastStep(nn.Module, SourceCodeLogger):

    def __init__(self, grid, dh=1,
                        encoder_hidden=384, H=192):
        super().__init__()


        self.grid = grid
        self.dh = dh
        self.input_dim = grid.n_pr_vars * grid.n_levels + grid.n_sfc_vars

        self.embed_dim = self.input_dim + 0
        self.D = 2*self.input_dim + self.grid.xpos.shape[-1]
        self.H = H
        transformer_input = H
        print("transformer_input_dim:", transformer_input)
        self.transformer = Transformer(dim=transformer_input, depth=16, heads=12, dim_head=32, mlp_dim=256)
        self.embedding = nn.Linear(self.D, self.H) 
        self.pos_embedding = nn.Parameter(torch.randn(1, self.grid.k, self.H)) #braindamaged

        self.decoder = nn.Sequential(
                nn.Linear(transformer_input, encoder_hidden),
                nn.SiLU(),
                nn.Linear(encoder_hidden, self.input_dim)
        )

        return

    def forward(self, x):
        B,N,k,D = x.shape

        x = torch.flatten(x, 0, 1) 
        x_emb = self.embedding(x)
        x_emb += self.pos_embedding
        y = self.transformer(x_emb)
        y = self.decoder(y)
        y = y.view(B, N, y.shape[1])

        return y

class ForecastStepLinear(nn.Module, SourceCodeLogger):

    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
        self.output_deltas = True
        self.forecast_dim = mesh.n_pr_vars * mesh.n_levels + mesh.n_sfc_vars 
        self.O = self.forecast_dim
        self.D = self.forecast_dim + 4
        self.linear = nn.Linear(self.D, self.O) 
        self.input_bias = nn.Parameter(torch.zeros(self.D))
        self.output_bias = nn.Parameter(torch.zeros(self.O))
        return

    def forward(self, x):
        assert type(self.mesh) is LatLonGrid, "Whoops, meshes are not supported now."
        B,N1,N2,D = x.shape
        x = torch.flatten(x, 0, 2)
        y = self.linear(x + self.input_bias) + self.output_bias
        y = y.view(B, N1, N2, self.O)
        #print(y.shape) 
        return y


