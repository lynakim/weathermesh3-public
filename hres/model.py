import torch
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import math
from pprint import pprint
from data import *
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

from torch.utils import checkpoint

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from matepoint import matepoint_pipeline
import matepoint as matepoint

Gmatepoint_stream = None

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

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
    def __init__(self, dim, heads = 8, dim_head = 64, silly_rel = False, seq_len=None, flash=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.silly_rel = silly_rel
        self.flash = flash

        if silly_rel:
            self.silly_bias = nn.Parameter(torch.zeros(heads, seq_len, seq_len))
            trunc_normal_(self.silly_bias, std=.02)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        if self.flash and 1:
            #assert not self.silly_rel
            if self.silly_rel:
                bb = self.silly_bias[None]
            else:
                bb = None
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                attn_v = torch.nn.functional.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), bb)

            #print("huh attn_v", attn_v.shape, "to_out", self.to_out)
            out = rearrange(attn_v, 'b h n d -> b n (h d)')
            #print("post rearrange", out.shape)
            return self.to_out(out)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            #print("huh dots", dots.shape)

            if self.silly_rel:
                dots += self.silly_bias[None]

            attn = self.attend(dots)

            out = torch.matmul(attn, v)
            #print("compare to attn", attn.shape, "out", out.shape)
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, silly_rel=False, seq_len=None, use_matepoint=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.use_matepoint = use_matepoint
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, silly_rel=silly_rel, seq_len=seq_len),
                FeedForward(dim, mlp_dim)
            ]))
        self.matepoint_ctx = []

    def forward(self, x):
        if self.use_matepoint:
            global Gmatepoint_stream
            Gmatepoint_stream = torch.cuda.Stream()
        for attn, ff in self.layers:
            if self.use_matepoint:
                x = x + matepoint.checkpoint(attn, x, matepoint_ctx=self.matepoint_ctx,stream=Gmatepoint_stream, use_reentrant=False, preserve_rng_state=False)
            else:
                x = x + checkpoint.checkpoint(attn, x, use_reentrant=False)
            #x = x + checkpoint.checkpoint(ff, x, use_reentrant=False)
            #x = attn(x) + x
            x = ff(x) + x
        matepoint_pipeline(self.matepoint_ctx,Gmatepoint_stream)
        return self.norm(x)


class HresModel(torch.nn.Module):
    def __init__(self, silly_rel=False, L=512, absbias=False, do_pressure=False, depth=12, use_matepoint=False, do_radiation=False, do_modis=True, postconv=True, pressure_vertical=4, sfc_extra=0, n_out=8, grid=""):
        super(HresModel, self).__init__()
        self.grid = grid
        self.sfc_extra = sfc_extra
        self.n_out = n_out
        assert not use_matepoint
        self.do_radiation = do_radiation
        self.do_modis = do_modis
        self.use_matepoint = use_matepoint
        self.do_pressure = do_pressure
        self.convs = nn.ParameterDict()
        self.L = L
        self.postconv = postconv

        if False:
            f = 2
            self.postconvs["mn30"] = nn.Sequential(nn.ReLU(), nn.Linear(f*L, L))
            self.postconvs["mn30"] = nn.Sequential(nn.ReLU(), nn.Linear(f*L, L))
        else:
            f = 1
        self.convs["mn30"] = nn.Conv2d(in_channels=3 + 17*do_modis, out_channels=f*L, kernel_size=(4,4), stride=(4,4))
        self.convs["mn75"] = nn.Conv2d(in_channels=3 + 17*do_modis, out_channels=f*L, kernel_size=(4,4), stride=(4,4))



        self.Lsfc = self.L
        if do_pressure: self.Lsfc = self.L // 2
        self.Lpr = self.L - self.Lsfc
        self.convs["sfc"] = nn.Conv2d(in_channels=9 + sfc_extra + self.do_radiation * 3, out_channels=self.Lsfc, kernel_size=(2,2), stride=(2,2))
        if self.do_pressure:
            last = 7 if pressure_vertical == 4 else 5
            self.convs["pr"] = nn.Sequential(
                nn.Conv3d(in_channels=5, out_channels=2*L, kernel_size=(pressure_vertical,2,2), stride=(pressure_vertical,2,2)),
                nn.ReLU(),
                nn.Conv3d(in_channels=2*L, out_channels=self.Lpr, kernel_size=(last,1,1), stride=(last,1,1)))

        if self.postconv:
            self.center_lin = nn.Sequential(
                nn.Linear(11 + sfc_extra + self.do_radiation*3 + do_modis * 2 * 17, 2*L),
                nn.ReLU(),
                nn.Linear(2*L, L)
            )
        else:
            self.center_lin = nn.Linear(11 + sfc_extra + self.do_radiation*3 + do_modis * 2 * 17, L)

        dims_per_head = 32
        heads = L // dims_per_head
        mlp_dim = L * 4

        self.output_lin = nn.Sequential(
            nn.Linear(L, 2*L),
            nn.ReLU(),
            nn.Linear(2*L, self.n_out)
        )

        seq_len = 193
        if self.grid == "_small":
            seq_len = 97

        self.transformer = Transformer(L, depth, heads, dims_per_head, mlp_dim, silly_rel=silly_rel, seq_len=seq_len, use_matepoint=self.use_matepoint)

        self.relcrap_idxs = None
        if absbias:
            self.absbias = nn.Parameter(torch.zeros(seq_len, L//4))
            trunc_normal_(self.absbias, std=.25)
        else:
            self.absbias = None

    def forward(self, dic):
        x = []
        x.append(self.center_lin(dic["center"])[:,:,None])
        
        grids = ["mn30", "mn75"]
        pos = []
        if self.do_pressure:
            sfc = torch.flatten(self.convs["sfc"](dic["sfc"]), start_dim=2)
            pr = torch.flatten(self.convs["pr"](dic["pr"]), start_dim=2)
            x.append(torch.cat((sfc, pr), dim=1))
        else:
            grids = ["sfc"] + grids

        for el in grids:
            x.append(torch.flatten(self.convs[el](dic[el]), start_dim=2))
        x = torch.cat(x, dim=2).permute(0, 2, 1)
        if self.absbias is not None:
            x[:, :, :self.absbias.shape[1]] += self.absbias[None]
        tr = self.transformer(x)
        out = self.output_lin(tr[:, 0, :])
        return out
        

if 0:
    arr = torch.randn(1,3,720,1440)
    lat = 45.25
    lon = 123.25
    lat, lon = 45.4, 123.6


    x = np.arange(90, -90, -0.25)
    y = np.arange(0, 360, 0.25)
    lons, lats= np.meshgrid(y, x)
    #print(lons)
    #print(lats)

    sint = RegularGridInterpolator((x, y), arr[0,0,:].numpy())

    allpts = [(np.random.uniform(low=-70., high=70), np.random.uniform(low=0., high=350.)) for _ in range(10)]
    Ref = []
    for lat, lon in allpts:
        pts = get_points(lat, lon, 20, 3)
        ref = []
        for a,b in pts:
            ref.append(sint([a,b])[0])
        Ref.append(ref)
    Ref = np.array(Ref)

    print("ref", Ref, Ref.shape)
    
    interp, idxs = get_interps(allpts, 20, 3)
    print("idxs", idxs.shape)
    print("interp", interp.shape)

    gath = arr[0,:, idxs[...,0], idxs[...,1]]
    print("gath", gath.shape)
    #vals = gath @ interp
    vals = torch.einsum('abcd,bcd->abc', gath, torch.FloatTensor(interp))[0]
    vals2 = (gath * torch.FloatTensor(interp)).sum(axis=3)[0]
    print(vals.shape, Ref.shape)
    print(torch.sqrt(torch.mean(torch.square(vals-torch.FloatTensor(Ref)))))
    print(torch.sqrt(torch.mean(torch.square(vals-vals2))))


    exit()


    #lat, lon = 80, 10
    ilat = int(round((90 - lat)*4))
    ilon = int(round(lon * 4))
    print("hey", ilat, ilon)

    tlat = int(math.floor((90-lat)*4))
    blat = min(tlat+1, 719)
    llon = int(math.floor(lon*4))
    rlon = (llon+1)%1440

    det = 1/(-0.25 * 0.25)
    v1 = np.array([(llon+1)*0.25 - lon, lon - llon*0.25])
    Arr = arr[0,0,:]
    Q = np.array([[Arr[tlat, llon], Arr[blat, llon]], [Arr[tlat, rlon], Arr[blat, rlon]]])
    print(Q)
    v2 = np.array([(90 - blat*0.25)-lat, lat - (90 - tlat*0.25)])
    print(v1)
    print(v2)
    idx = [(tlat, llon), (blat, llon), (tlat, rlon), (blat, rlon)]
    vals = det * np.outer(v1, v2).flatten()
    #print("huh", arr[0,0,idx].shape)
    huh = arr[0,:, [x[0] for x in idx], [x[1] for x in idx]]
    print("hey", huh.shape)
    print("interp", huh@vals)

    interp = det * v1.T @ Q @ v2
    print("npref", interp)

    grid = torch.zeros(1, 1, 1, 2)
    grid[0,0,0][1]  = -lat/90
    grid[0,0,0][0] = (lon-180)/180
    print("uhhh", grid)
    tt = torch.nn.functional.grid_sample(arr, grid, mode='bilinear', align_corners=True)
    print(tt)
    print("ref", arr[0, :, ilat, ilon])
    print("scipy", sint([lat, lon]))
