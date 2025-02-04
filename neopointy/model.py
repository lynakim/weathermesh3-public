import torch
import time
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import math
from pprint import pprint
from data import *
import torch.nn as nn

from hres_utils import *

from einops import rearrange
from einops.layers.torch import Rearrange


from natten.functional import na2d
from torch.utils import checkpoint

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from matepoint import matepoint_pipeline
import matepoint as matepoint

from natten.functional import na2d_av, na2d_qk
from natten.types import (
    CausalArg2DTypeOrDed,
    Dimension2DTypeOrDed,
    CausalArg3DTypeOrDed,
    Dimension3DTypeOrDed,
    FnaBackwardConfigType,
    FnaForwardConfigType,
    NoneType,
)
from natten.utils import (
    check_additional_keys,
    check_additional_values,
    check_all_args,
    check_backward_tiling_config,
    check_tiling_config,
    get_num_na_weights,
    log,
    make_attn_tensor_from_input,
)

from torch import Tensor
from typing import Any, Optional, Tuple
from natten.functional import FusedNeighborhoodAttention3D
from natten.functional import FusedNeighborhoodAttention2D

Gmatepoint_stream = None

def call_checkpointed(module, *args, enabled=True, no_matepoint=False, **kwargs):
    #import traceback
    #traceback.print_stack()
    # Making a function for this because I routinely forget the syntax and would be confused about what paramters I want to set

    if not enabled: return module(*args, **kwargs) # so that it's easy to disable checkpointing for debugging 
    if no_matepoint: return torch.utils.checkpoint.checkpoint(module, *args, **kwargs, use_reentrant=False, preserve_rng_state=False)

    # The cude stream needs to be created only once per process. This handles this for you.
    #
    # NOTE: we used to have these be global in the model files, but it caused some annoying things. I realized that I think it works being global in the 
    # matepoint file, but you have to use import matepoint and then matepoint.Gmatepoint_stream and stuff. Otherwise, it imports a reference. Ask an LLM if you are confused.

    #if matepoint.Gmatepoint_stream is None:
    #    matepoint.Gmatepoint_stream = torch.cuda.Stream()

    return matepoint.checkpoint(module, *args, **kwargs, use_reentrant=False, preserve_rng_state=False, stream=matepoint.Gmatepoint_stream)

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

from typing import Optional, Tuple
os.environ['NATTEN_LOG_LEVEL'] = 'error'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #for annoying natten warnings

import natten
from natten.utils import check_all_args
natten.use_fused_na(True)
natten.use_kv_parallelism_in_fused_na(True)
natten.set_memory_usage_preference("unrestricted")

#natten.use_kv_parallelism_in_fused_na(False)
#natten.set_memory_usage_preference("strict")

natten.use_tiled_na()
natten.use_gemm_na()

def tuned_na3d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension3DTypeOrDed,
    dilation: Dimension3DTypeOrDed = 1,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
    rpb: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> Tensor:
    if query.is_nested or key.is_nested or value.is_nested:
        raise NotImplementedError(
            "Fused neighborhood attention does not support nested tensors yet."
        )

    #tiling_config_forward, tiling_config_backward = autotune_fna(
    #    3, query, kernel_size, dilation, is_causal
    #)
    tiling_config_forward, tiling_config_backward = ((8, 2, 4), (8, 2, 4)), ((8, 4, 2), (8, 2, 4), (1, 45, 30), False)
    scale = scale or query.shape[-1] ** -0.5

    return FusedNeighborhoodAttention3D.apply(
        query,
        key,
        value,
        rpb,
        kernel_size,
        dilation,
        is_causal,
        scale,
        tiling_config_forward,
        tiling_config_backward,
    )

def tuned_na2d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension2DTypeOrDed,
    dilation: Dimension2DTypeOrDed = 1,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
    rpb: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> Tensor:
    if query.is_nested or key.is_nested or value.is_nested:
        raise NotImplementedError(
            "Fused neighborhood attention does not support nested tensors yet."
        )

    #print("yo", query.shape, key.shape, value.shape)
    tiling_config_forward, tiling_config_backward = ((8, 8), (8, 8)), ((8, 8), (4, 16), (90, 22), False)
    tiling_config_forward, tiling_config_backward = ((8, 8), (8, 8)), ((8, 8), (4, 16), (90, 3), False)
    if query.shape[1] < 300:
        tiling_config_forward, tiling_config_backward = ((8, 8), (8, 8)), ((8, 8), (8, 8), (1, 1), False)
    scale = scale or query.shape[-1] ** -0.5

    return FusedNeighborhoodAttention2D.apply(
        query,
        key,
        value,
        rpb,
        kernel_size,
        dilation,
        is_causal,
        scale,
        tiling_config_forward,
        tiling_config_backward,
    )




class CustomNeighborhoodAttention2D(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int | Tuple[int, int, int],
        dilation: int | Tuple[int, int, int] = 1,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        assert dilation == 1

        super().__init__()
        kernel_size, dilation, is_causal = check_all_args(
            2, kernel_size, dilation, False
        )

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"NeighborhoodAttention2D expected a rank-4 input tensor; got {x.dim()=}."
            )

        B, H, W, C = x.shape


        qkv = (
            self.qkv(x)
            .reshape(B, H, W, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 1, 2, 4, 5)
        )
        #    .permute(4, 0, 5, 1, 2, 3, 6) # this is for non FNA
        q, k, v = qkv[0], qkv[1], qkv[2]
        #q = q * self.scale

        #assert natten.context.is_fna_enabled()
        #print("rpb shape", self.rpb.shape, "qkv", qkv.shape, q.shape, k.shape, v.shape, "x", x.shape)
        # TODO look at q scale
        x_2 = tuned_na2d(
            q,
            k,
            v,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            is_causal=False,
            rpb=None,
        )
        """
        attn_ref = na2d_qk(
            q,
            k,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            is_causal=False,
            rpb=None,
        )
        attn_ref = attn_ref.softmax(dim=-1)
        x_2 = na2d_av(
            attn_ref, v, kernel_size=self.kernel_size, dilation=self.dilation, is_causal=False)
        """
        
        #print("got", out_ref.shape, B, H, W, C)
        #x_2 = x_2.permute(0, 4, 1, 2, 3, 5)
        """
        0 1 2 3 4 5
        0 4 1 2 3 5
        0 1 2 3 4 5 
        """
        x = x_2.reshape(B, H, W, C)

        return self.proj_drop(self.proj(x))



class CustomNeighborhoodAttention3D(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int | Tuple[int, int, int],
        dilation: int | Tuple[int, int, int] = 1,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        assert dilation == 1

        super().__init__()
        kernel_size, dilation, is_causal = check_all_args(
            3, kernel_size, dilation, False
        )

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"NeighborhoodAttention2D expected a rank-5 input tensor; got {x.dim()=}."
            )

        B, D, H, W, C = x.shape


        qkv = (
            self.qkv(x)
            .reshape(B, D, H, W, 3, self.num_heads, self.head_dim)
            .permute(4, 0, 1, 2, 3, 5, 6)
        )
        #    .permute(4, 0, 5, 1, 2, 3, 6) # this is for non FNA
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #assert natten.context.is_fna_enabled()
        #print("rpb shape", self.rpb.shape, "qkv", qkv.shape, q.shape, k.shape, v.shape, "x", x.shape)
        # TODO look at q scale
        x_2 = tuned_na3d(
            q,
            k,
            v,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            is_causal=False,
            rpb=None,
        )
        #x_2 = x_2.permute(0, 4, 1, 2, 3, 5)
        """
        0 1 2 3 4 5
        0 4 1 2 3 5
        0 1 2 3 4 5 
        """
        x = x_2.reshape(B, D, H, W, C)

        return self.proj_drop(self.proj(x))
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    

class Natten3DTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=None, 
                 mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        assert window_size is not None

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)        
        self.attn = CustomNeighborhoodAttention2D(
            dim,
            num_heads,
            window_size,
        )

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):

        #torch.cuda.empty_cache()
        #mem = torch.cuda.mem_get_info()
        #print("3dtransformerblock", "mem", mem[0]/(1024**3), mem[1]/(1024**3))
        
        #B, D, H, W, C = x.shape
        B, H, W, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        #torch.cuda.empty_cache()
        #mem = torch.cuda.mem_get_info()
        #print("    post", "mem", mem[0]/(1024**3), mem[1]/(1024**3))

        return x
    
class SlideLayers3D(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size

        mlist = []
        for _ in range(depth):
            tb_block = Natten3DTransformerBlock(
                dim,
                num_heads,
                window_size=window_size,
            )
            mlist.append(tb_block)
        self.blocks = nn.ModuleList(mlist)

    def forward(self, x):
        #B, D, H, W, C = x.shape
        B, H, W, C = x.shape

        for _, blk in enumerate(self.blocks):
            x = call_checkpointed(blk, x)

        return x


class HresModel(torch.nn.Module):
    def __init__(self, steps, n_out=8, initial_L=64, chunk_size=None, dims_per_head=16, do_pr=False):
        super(HresModel, self).__init__()
        self.n_out = n_out
        self.chunk_size = chunk_size

        self.initial_L = initial_L
        self.convs = nn.ParameterList()
        self.trans = nn.ParameterList()
        self.avgpool = nn.ParameterList()
        self.maxpool = nn.ParameterList()
        self.encpooled = nn.ParameterList()
        total_fac = np.prod([x[0] for x in steps])
        cumfac = 1
        initial_L = 8 + 1 + 2 + 5
        prev_L = initial_L
        for i, step in enumerate(steps):
            factor, L, n_trans, wsize = step
            self.convs.append(nn.ConvTranspose2d(in_channels=prev_L, out_channels=L, kernel_size=(factor, factor), stride=(factor, factor)))
            self.trans.append(SlideLayers3D(dim=L, depth=n_trans, num_heads=L//dims_per_head, window_size=(wsize, wsize)))
            cumfac *= factor
            ff = int(total_fac//cumfac)
            assert total_fac % cumfac == 0
            self.avgpool.append(torch.nn.AvgPool2d(kernel_size=(ff, ff)))
            self.maxpool.append(torch.nn.MaxPool2d(kernel_size=(ff, ff)))
            self.encpooled.append(nn.Conv2d(in_channels=4 + 17 + initial_L, out_channels=L, kernel_size=1))
            prev_L = L
        self.steps = steps

        #self.output = nn.Linear(L, n_out)
        # L is the L from the last transformer
        self.output_lin = nn.Sequential(
            nn.Linear(L, 2*L),
            nn.ReLU(),
            nn.Linear(2*L, self.n_out)
        )

    def convcrap(self, i, x, pooled):
        encoded = self.encpooled[i](pooled)
        posemb = posemb_sincos_2d(encoded.shape[2], encoded.shape[3], encoded.shape[1], dtype=torch.float16).to(encoded.device)
        posemb = posemb.view(encoded.shape[2], encoded.shape[3], encoded.shape[1]).permute(2, 0, 1)

        x = x.permute(0, 3, 1, 2)
        main = self.convs[i](x)
        main = main + encoded + posemb[None]
        x = main.permute(0, 2, 3, 1)
        return x


    def forward(self, dic):
        x = torch.cat((dic["era5sfc"], dic["Rad"], dic["Ang"], dic["Static_sfc"]), axis=-1)
        initial = x.clone().permute(0,3,1,2)
        for i, step in enumerate(self.steps):
            """
            torch.cuda.empty_cache()
            mem = torch.cuda.mem_get_info()
            print("starting step", i, "mem", mem[0]/(1024**3), mem[1]/(1024**3))
            t0 = time.time()
            """
            factor, L, n_trans, wsize = step

            elev = self.avgpool[i](dic["Elevation30"].permute(0, 3, 1, 2))
            modis = self.maxpool[i](dic["Modis30"].half())
            onehot = torch.nn.functional.one_hot(modis.long(), 17).permute(0, 3, 1, 2).half()
            #print("step",i,"factor", elev.shape[2]//initial.shape[2], "initial.shape", initial.shape, "elev", elev.shape)
            upscaled_initial = torch.nn.functional.interpolate(initial, scale_factor=elev.shape[2]//initial.shape[2], align_corners=True, mode='bilinear')
            pooled = torch.cat((elev, onehot, upscaled_initial), dim=1)
            #print("sz", pooled.nbytes/1e6, modis.nbytes/1e6, elev.nbytes/1e6, "x", x.nbytes/1e6)
            del onehot, elev, modis
            x = call_checkpointed(self.convcrap, i, x, pooled)
            #x = self.convcrap(i, x, pooled)
            del pooled

            if self.chunk_size is not None:
                cs = self.chunk_size
                nc = x.shape[0] // cs
                xx = []
                for j in range(nc):
                    xx.append(self.trans[i](x[cs*j:cs*(j+1)]))
                x = torch.cat(xx, axis=0)
                del xx
            else:
                x = self.trans[i](x)

            """
            dt = time.time()-t0
            torch.cuda.empty_cache()
            mem = torch.cuda.mem_get_info()
            print("finished step", i, "mem", mem[0]/(1024**3), mem[1]/(1024**3), "took", dt)
            """
        #out = self.output_lin(x)
        out = call_checkpointed(self.output_lin, x)
        #print("out shape", out.shape)
        return out
        exit()
        import pdb
        pdb.set_trace()


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
