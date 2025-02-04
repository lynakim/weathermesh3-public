import socket, os
import psutil
from utils import *
from data import *
import gc
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from einops import rearrange
import matepoint as matepoint
from matepoint import matepoint_pipeline
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import warnings
warnings.filterwarnings("ignore",category=FutureWarning, message=".*torch.cuda.amp.custom.*")

torch.backends.cudnn.benchmark = False

from natten.types import (
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


def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    #print("uhhh", patches.shape, patches.device, patches.dtype)
    #print("er patches", patches)
    (_, f, h, w, dim), device, dtype = patches#*patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij')

    fourier_dim = dim // 6
    #print("Hey dim is", dim, "fourier", fourier_dim)

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)


timer0 = 0
Gmatepoint_stream = None


class ForecastStepConfig():
    def __init__(self,inputs,**kwargs):
        self.inputs = inputs
        self.outputs = inputs
        self.patch_size = (4,8,8)
        self.preconv_hidden_dim = 128
        self.hidden_dim = 896
        self.FLASH = True
        self.dims_per_head = 32
        self.drop_path = 0.0
        self.skip_every = None
        self.window_size = (3,5,7)
        self.delta_processor = False
        self.activation = nn.GELU()
        self.sincos = True
        self.Encoder = ForecastStepEncoder
        self.Decoder = ForecastStepDecoder
        self.ucodec_config = UCodecConfig()
        self.Transformer = SlideLayer3D
        self.adapter_swin_depth = 0
        self.padded_lon = True
        self.enc_swin_depth = 4
        self.proc_swin_depth = 8
        self.dec_swin_depth = 4
        self.proc_depths = None
        self.timesteps = [24]
        self.train_timesteps = None
        self.processor_dt = 6
        self.return_random = False
        self.parallel_encoders = False
        self.rollout_reencoder = False
        self.rollout_inputs = None
        self.output_deltas = False
        self.checkpointfn = matepoint.checkpoint
        self.checkpoint_convs = False

        self.decoder_reinput_initial = False
        self.decoder_reinput_size = 0
        self.perturber = 0
        self.adapter_H1 = 16
        self.adapter_dim_seq = [64,128,256]
        self.adapter_use_input_bias = not True
        self.load_half = True
        self.neorad = True
        self.neorad_subsamp = True
        self.n_const_vars = 12
        self.sincos_latlon = False
        self.name = ''

        self.matepoint_table = False # options: False, 'auto, number

        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a ForecastStepConfig attribute"
            assert k != 'timesteps', "You no longer can configure timesteps explicitly, it's all just based on the processor"
            setattr(self,k,v)
        self.update()

    def update(self): 
        if self.train_timesteps is None:
            self.train_timesteps = self.timesteps
        self.mesh = self.inputs[0]
        self.output_mesh = self.outputs[0]
        #assert self.mesh.extra_sfc_vars == self.output_mesh.extra_sfc_vars[:len(self.mesh.extra_sfc_vars)] or self.mesh.extra_sfc_vars == ['zeropad', '45_tcc', '034_sstk', '168_2d', 'zeropad', 'zeropad', 'zeropad', 'zeropad', 'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'], "extra_sfc_vars of input must match output (allowing for more output-only variables at the end)"

        assert self.processor_dt is not None
        assert isinstance(self.processor_dt,int) or isinstance(self.processor_dt,list), "processor_dt must be int or list"
        if isinstance(self.processor_dt,int): self.processor_dt = [self.processor_dt]
        assert 0 not in self.processor_dt
        self.processor_dt.sort()

        if self.rollout_reencoder:
            assert self.rollout_inputs is not None
        for t in self.timesteps:
            if self.processor_dt == 0: assert t == 0, "timesteps must be 0 if processor_dt is 0"
            elif isinstance(self.processor_dt,int):
                assert t % self.processor_dt  == 0, f"processor_dt {self.processor_dt} must divide all timesteps {self.timesteps}"
            else:
                at_least_one = False
                for pdt in self.processor_dt:
                    if t % pdt == 0:
                        at_least_one = True
                        break
                assert at_least_one, f"at least one processor_dt must divide all timesteps {self.timesteps}"
        assert self.hidden_dim % self.dims_per_head == 0
        self.num_heads = self.hidden_dim // self.dims_per_head

        self.resolution = ((self.mesh.n_levels)//self.patch_size[0] + 1, self.mesh.lats.shape[0] // self.patch_size[1], 
            self.mesh.lons.shape[0] // self.patch_size[2] + 2 * (self.window_size[2]//2)*self.padded_lon) 
 
        self.latent_len = self.hidden_dim * self.resolution[0] * self.resolution[1] * self.resolution[2]
        self.latent_size = self.latent_len * 2

        if self.sincos_latlon:
            self.n_const_vars += 2
        if self.neorad: self.n_addl_vars = 4
        else: self.n_addl_vars = 2
        
        self.total_sfc = self.mesh.n_sfc_vars + self.n_const_vars + self.n_addl_vars

        print(ORANGE(f"Latent size: {self.latent_size / 2**20:0.2f} MiB"),only_rank_0=True)

        if self.Encoder.__name__ == 'ForecastStepUEncoder':
            assert self.Decoder.__name__ == 'ForecastStepUDecoder'
            uc = self.ucodec_config
            ns_in = self.total_sfc * uc.conv_sz[0][1] * uc.conv_sz[0][2]
            ns_cout = 2*uc.conv_dim[0]
            print(ORANGE(f"UCodec conv_sfc: out/in = {ns_cout}/{ns_in} = {ns_cout/ns_in}"),only_rank_0=True)

            np_in = self.mesh.n_pr_vars * uc.conv_sz[0][0] * uc.conv_sz[0][1] * uc.conv_sz[0][2]
            np_cout = uc.conv_dim[0]
            print(ORANGE(f"UCodec conv_pr: out/in = {np_cout}/{np_in} = {np_cout/np_in}"),only_rank_0=True)

            for i in range(len(uc.conv_dim)-1):
                c_in = uc.conv_dim[i] * uc.conv_sz[i+1][0] * uc.conv_sz[i+1][1] * uc.conv_sz[i+1][2]
                c_out = uc.conv_dim[i+1]
                print(ORANGE(f"UCodec conv[{i}]: out/in = {c_out}/{c_in} = {c_out/c_in}"),only_rank_0=True)

        



        #self.resolution = (self.mesh.n_levels//self.patch_size[0] + 1 + 1, self.mesh.lats.shape[0] // self.patch_size[1], 
        #    self.mesh.lons.shape[0] // self.patch_size[2])  #XXX HERE!!

        ## legacy shit

        assert isinstance(self.processor_dt, int) or isinstance(self.processor_dt, list), "processor_dt must be an int or a list"
        if isinstance(self.processor_dt, int):
            self.processor_dt = [self.processor_dt]
        else:
            self.processor_dt = self.processor_dt

        if self.proc_depths == None:
            self.proc_depths = [self.proc_swin_depth] * len(self.processor_dt)


        self.timesteps = range(0,1000,self.processor_dt[0]) # fuckit why can't our models predict 1000 hours into the future?
        self.conv_dim = self.hidden_dim
        self.transformer_dim = self.hidden_dim



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        #self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.drop(x)
        return x


from typing import Optional, Tuple
os.environ['NATTEN_LOG_LEVEL'] = 'error'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #for annoying natten warnings
from torch import nn, Tensor
try:
    import natten
    # assert natten.WB_MODDED 
    from natten.functional import na3d, na3d_av, na3d_qk
    from natten.utils import check_all_args
    natten.use_fused_na(True)
    #natten.set_memory_usage_preference("unrestricted")
    natten.use_kv_parallelism_in_fused_na(True)
    natten.set_memory_usage_preference("unrestricted")
except:
    builtins.print("natten not installed, skipping import")
    pass

#natten.use_fused_na(False)
#os.environ['NATTEN_LOG_LEVEL'] = 'debug'
"""natten.use_autotuner(
  backward_pass=True,
  thorough_mode_backward=True,
)
"""
"""
natten.use_autotuner(
  forward_pass=True,
  backward_pass=True,
  warmup_steps_forward=10,
  warmup_steps_backward=20,
  steps_forward=5,
  steps_backward=10,
)
"""
# natten.use_autotuner(forward_pass=True, backward_pass=True)
# ((2, 4, 4), (4, 8, 4)), ((2, 8, 4), (2, 4, 8), (3, 30, 30), False)


class CustomNeighborhoodAttention3D(nn.Module):
    """
    Neighborhood Attention 3D Module
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int | Tuple[int, int, int],
        dilation: int | Tuple[int, int, int] = 1,
        is_causal: bool | Tuple[bool, bool, bool] = False,
        rel_pos_bias: bool = False,
        earth_pos_bias: bool = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        world_size = None # what the fuck is with the typing
    ):
        assert dilation == 1
        assert not is_causal
        assert world_size is not None
        self.world_size = world_size

        super().__init__()
        kernel_size, dilation, is_causal = check_all_args(
            3, kernel_size, dilation, is_causal
        )
        if any(is_causal) and rel_pos_bias:
            raise NotImplementedError(
                "Causal neighborhood attention is undefined with positional biases."
                "Please consider disabling positional biases, or open an issue."
            )

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.is_causal = is_causal

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
            is_causal=self.is_causal,
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

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, "
            + f"dilation={self.dilation}, "
            + f"is_causal={self.is_causal}, "
        )


class Natten3DTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=None, 
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_mask = None, FLASH=True):
        super().__init__()
        assert window_size is not None
        #assert window_size[1] == window_size[2], "For now need square windows, SAD!"
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        #for a,b in zip(self.input_resolution, self.window_size):
            #assert a >= b
            #assert a % b == 0

        self.norm1 = norm_layer(dim)        
        self.attn = CustomNeighborhoodAttention3D(
            dim,
            num_heads,
            window_size,
            earth_pos_bias=False,
            rel_pos_bias=True,
            world_size=input_resolution
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None):
        Z, H, W = self.input_resolution
        B, L, C = x.shape
        #print("uh oh BLC", B, L, C, "L", L, "ZHW", Z, H, W)
        assert L == Z * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Z, H, W, C)
        x = self.attn(x)
        x = x.view(B, Z * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SlideLayer3D(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, config, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 fused_window_process=False, FLASH=True, skip_every=None,perturber=0,padded_lon=False,
                 checkpoint_chunks=1
                 ):

        super().__init__()
        self.config = config
        self.padded_lon = padded_lon
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.skip_every = None
        self.window_size = window_size
        self.checkpoint_chunks = checkpoint_chunks
        resolution = input_resolution
        if checkpoint_chunks > 1:
           assert checkpoint_chunks == 2, "there is annoying shit needed if you want more than 2"
           assert self.config.checkpointfn is not None
           assert input_resolution[1] % checkpoint_chunks == 0
           assert input_resolution[2] % checkpoint_chunks == 0

           winpad = window_size[2]//2 * 2
           # the transformer blocks will take in a chunk of the input at a time. Chunks need to be padded, and we need to subtract the lon pading if it exists.

           resolution = (input_resolution[0], input_resolution[1], (input_resolution[2])//checkpoint_chunks + winpad)
           self.chunk_resolution = resolution   
            

        mlist = []
        for i in range(depth):
            tb_block = Natten3DTransformerBlock(
                dim,
                resolution,
                num_heads,
                window_size=window_size,
            )


            """
            tb_block = CustomNeighborhoodAttention3D(
                dim=dim,
                kernel_size=window_size[1],
                kernel_size_d=window_size[0],
                dilation=1,
                dilation_d=1,
                num_heads=num_heads,
                Mlev=input_resolution[0],
                Mlat=input_resolution[1],
                Mlon=input_resolution[2])
            Swin3DTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=tuple([0 if (i % 2 == 0) else x // 2 for x in window_size]),
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process, FLASH=FLASH)
            """
            mlist.append(tb_block)
        self.blocks = nn.ModuleList(mlist)

    def forward(self, x):
        global Gmatepoint_stream
        assert self.skip_every is None
        
        B, fL, C = x.shape
        fD, fH, fW = self.input_resolution
        if self.checkpoint_chunks > 1: 
            cD, cH, cW = self.chunk_resolution
            cL = cD*cH*cW

        for i, blk in enumerate(self.blocks):
            #plt.imsave('ohp1.png',x.view(B, *self.input_resolution, C)[0,0,:,:,5].cpu().detach().numpy())
            if self.padded_lon:
                pad = (self.window_size[2]//2)
                w = self.input_resolution[2] - 2*pad
                xd = x.view(B, *self.input_resolution, C)
                xd[:, :, :, :pad, :] = xd[:, :, :, w-pad:w].clone()
                xd[:, :, :, pad+w:, :] = xd[:, :, :, pad:pad+pad].clone()
                x = xd.view(*x.shape)
                del xd
            pass
            # plt.imsave('ohp.png',x.view(B, *self.input_resolution, C)[0,0,:,:,5].cpu().detach().numpy())
            if self.checkpoint_chunks > 1:
                nc = self.checkpoint_chunks
                assert fW % nc == 0, f'fW {fW} must be divisible by {nc}'
                pad = (self.window_size[2]//2)
                assert pad != 0 
                xn = torch.zeros_like(x) # could save ram by not making these new tensors, but likely not worthwhile.
                for j in range(nc):
                    w0 = j     * fW//nc; w0p = w0 - pad*(j!=0)
                    w1 = (j+1) * fW//nc; w1p = w1 + pad*(j+1!=nc)
                    xcl = pad*(j==0)
                    xcr =  self.chunk_resolution[2]-pad*(j+1==nc)
                    xc = torch.zeros(B, *self.chunk_resolution, C, device=x.device)
                    xc[:,:,:,xcl:xcr,:] = x.view(B,fD,fH,fW,C)[:, :, :, w0p:w1p,:]
                    xco = call_checkpointed(self,blk,xc.view(B,cL,C))
                    xn.view(B,fD,fH,fW,C)[:,:,:,w0:w1,:] = xco.view(B,cD,cH,cW,C)[:,:,:,pad:-pad]
                x = xn

            else:
                x = call_checkpointed(self,blk, x)

        """
        if socket.gethostname() == "stinson":# and int(os.environ.get("LOCAL_RANK")) == 0:
            print(len(self.matepoint_ctx), "ram", psutil.Process(os.getpid()).memory_info()[0]/1e9, "GB", os.getpid())
            last = None
            for y, _, _ in self.matepoint_ctx[-6:]:
                a = y[0]
                ip = int(a.data_ptr())
                print(len(y), "dataptr", a.data_ptr(), a.shape, a.dtype, a.stride(), a.element_size(), a.nbytes/1e6, "MB", "pinned?", a.is_pinned())
                if last is not None:
                    print("delta", (ip-last)/(2**30), "GiB")
                last = ip
            """



        return x


def call_checkpointed(self, module, x):

    checkpointfn = self.config.checkpointfn
    if checkpointfn is not None:
        if checkpointfn == matepoint.checkpoint: x = checkpointfn(module, x, matepoint_ctx=self.config.matepoint_ctx,stream=Gmatepoint_stream, use_reentrant=False, preserve_rng_state=False)
        else:
            x = checkpointfn(module, x, use_reentrant=False, preserve_rng_state=False)
    else:
        x = module(x)

    return x

class EarthSpecificModel(torch.nn.Module):
    def __init__(self, config):
        c = config; mesh = c.mesh
        super().__init__()
        const_vars = []
        self.n_const_vars = 0
        to_cat = []
        if c.sincos_latlon == True:
            latlon = torch.FloatTensor(mesh.xpos)
            slatlon = torch.sin((latlon*torch.Tensor([np.pi/2,np.pi])))
            clatlon = torch.cos((latlon*torch.Tensor([np.pi/2,np.pi])))
            const_vars += ['sinlat','sinlon','coslat','coslon']; self.n_const_vars += 4; to_cat += [slatlon,clatlon]
        else:
            latlon = torch.FloatTensor(mesh.xpos)
            const_vars += ['lat','lon']; self.n_const_vars += 2; to_cat += [latlon]

        land_mask_np = np.load(CONSTS_PATH+'/land_mask.npy')
        land_mask = torch.BoolTensor(np.round(self.downsample(land_mask_np, mesh.xpos.shape)))
        const_vars += ['land_mask']; self.n_const_vars += 1; to_cat += [land_mask.unsqueeze(-1)]

        soil_type_np = np.load(CONSTS_PATH+'/soil_type.npy')
        soil_type_np = self.downsample(soil_type_np, mesh.xpos.shape,reduce=np.min)
        soil_type = torch.BoolTensor(self.to_onehot(soil_type_np))
        const_vars += ['soil_type']; self.n_const_vars += soil_type.shape[-1]; to_cat += [soil_type]

        elevation_np = np.load(CONSTS_PATH+'/topography.npy')
        elevation_np = self.downsample(elevation_np, mesh.xpos.shape,reduce=np.mean)
        elevation_np = elevation_np / np.std(elevation_np)
        elevation = torch.FloatTensor(elevation_np)
        const_vars += ['elevation']; self.n_const_vars += 1; to_cat += [elevation.unsqueeze(-1)]

        const_data = torch.cat(to_cat, axis=-1)
        self.register_buffer('const_data', const_data, persistent=False)
        assert self.n_const_vars == c.n_const_vars, f"{self.n_const_vars} vs {c.n_const_vars}"  
        self.const_vars = const_vars
        #todo: make different data types to save ram maybe
        # also a model actually has mutliple of these, could combine to save ram

    @staticmethod
    def downsample(mask,shape,reduce=np.mean):
        dlat = (mask.shape[0]-1) // shape[0]
        dlon = mask.shape[1] // shape[1]
        assert dlon == dlat
        d = dlat
        toshape = (shape[0], d, shape[1], d)
        #fuck the south pole
        ret = reduce(mask[:-1,:].reshape(toshape),axis=(1,3)) 
        assert ret.shape == shape[:2], (ret.shape, shape[:2])
        return ret

    @staticmethod 
    def to_onehot(x):
        x = x.astype(int)
        D = np.max(x)+1
        return np.eye(D)[x]


class ForecastStepBase(EarthSpecificModel,SourceCodeLogger):

    def __init__(self,config):
        self.mesh = config.mesh
        super().__init__(config)
        self.config = config
        self.radiation_cache = {}
        self.input_dim = self.mesh.n_pr_vars * self.mesh.n_levels + self.mesh.n_sfc_vars
        self.O = self.input_dim
        self.D = self.input_dim + self.config.n_addl_vars + self.n_const_vars
        self.total_sfc = self.mesh.n_sfc_vars + self.config.n_addl_vars + self.n_const_vars
        assert self.total_sfc == self.config.total_sfc, f"total_sfc {self.total_sfc} must match config {self.config.total_sfc}" 
        self.total_pr = self.mesh.n_pr_vars * self.mesh.n_levels
        self.name = ''
        self.neocache = {}


    def get_matepoint_tensors(self):
        tots = []
        for name, module in self.named_modules():
            if hasattr(module, 'matepoint_ctx'):
                #print(f"matepoint_ctx in {name}, len {len(module.matepoint_ctx)}")
                tensors = [x[0] for xx in module.matepoint_ctx for x in xx if isinstance(x[0], torch.Tensor)]
                tots += tensors
        return tots


    def interpget(self, src, toy, hr, a=300, b=400):
        if src not in self.neocache:
            self.neocache[src] = {}

        def load(xx, hr):
            if (xx,hr) in self.neocache[src]:
                return self.neocache[src][(xx, hr)]
            #print("loading", src, xx, hr)
            while True:
                if os.path.exists(CONSTS_PATH):
                    break
                print(f"Waiting for /fast to come online") # doesn't get printed for some reason
                time.sleep(10)
            F = torch.HalfTensor if self.config.load_half else torch.Tensor
            ohp = F(((np.load(CONSTS_PATH+'/%s/%d_%d.npy' % (src, xx, hr)) - a) / b))
            self.neocache[src][(xx,hr)] = ohp
            return ohp
        if self.config.neorad_subsamp and toy % 2 == 1:
            avg = load((toy-1)%366, hr) * 0.5 + load((toy+1)%366, hr) * 0.5
        else:
            avg = load(toy, hr)
        return avg

    
    def get_radiation(self, toy, hr=0):
        return self.interpget("neoradiation_%d"%self.mesh.subsamp, toy, hr)

    def add_additional_vars(self,x,t0s):
        
        # This fucntion adds additional variables to the input tensor which are not dependent on the input data,
        # but do change vs time. This is different than the additional variables which are constant vs time, those are 
        # done in the EarchSpecificModel class.

        dates = [get_date(t0.item()) for t0 in t0s]
        soys = [date.replace(month=1, day=1) for date in dates]
        #soy = date.replace(month=1, day=1)
        toys = [int((date - soy).total_seconds()/86400) for date, soy in zip(dates, soys)]
        #toy = int((date - soy).total_seconds()/86400)

        radiations = [self.get_radiation(toy, date.hour)[:720//self.mesh.subsamp,:,np.newaxis].unsqueeze(0).to(x.device) for toy, date in zip(toys, dates)]
        radiations = torch.cat(radiations, dim=0)
        #print("hey radiations shapes", radiations.shape)
        hours = torch.tensor([date.hour/24 for date in dates])
        #print("heyo hours shapes", hours.shape)
        timeofday = (torch.zeros_like(radiations, device=hours.device) + hours[:,None, None, None]).to(x.device)
        if self.config.neorad:
            try:
                angs = [self.interpget("solarangle_%d"%self.mesh.subsamp, toy, date.hour, a=0, b=180/np.pi) for toy, date in zip(toys, dates)]
                angs = [ang[:720//self.mesh.subsamp, :, np.newaxis].unsqueeze(0) for ang in angs]
                angs = torch.cat(angs, dim=0).to(x.device)

                sa = torch.sin(angs)
                ca = torch.cos(angs)
                xx = sa.float()
                #print("hiya mean", torch.mean(xx), "std", torch.std(xx), "neoradmean", torch.mean(radiation), "date", toy, date.hour)
                #print("yooo ang", ang.shape)
                x = torch.cat((x, radiations, timeofday, sa, ca), axis=-1)
                del angs
                del sa
                del ca
                del radiations
                del timeofday
            except Exception as e:
                raise e
                print(f"Failed to load solarangle_{self.mesh.subsamp} for {toy} {date} {date.hour}", flush=True)
                raise e
        else:
            x = torch.cat((x, radiations, timeofday), axis=-1)
        return x


    def breakup_pr_sfc(self,x,t0s):
        assert x.shape[-1] in [len(m.full_varlist) for m in self.config.inputs], f"X shape bad {x.shape} {[len(m.full_varlist) for m in self.config.inputs]}"
        c = self.config
        x = self.add_additional_vars(x, t0s)

        B,Nlat,Nlon,D = x.shape
        assert self.total_pr + self.total_sfc  - self.n_const_vars == D, f"{x.shape} c.total_pr {self.total_pr} c.total_sfc {self.total_sfc} c.n_const_vars {self.n_const_vars}"
        #assert self.total_pr + self.total_sfc == D
        xpr = x[:,:,:, :self.total_pr]
        xpr = xpr.view(B, Nlat, Nlon, c.mesh.n_pr_vars, c.mesh.n_levels)
        xpr = xpr.permute(0, 3, 4, 1, 2)

        xsfc = x[:,:,:, self.total_pr:]
        xsfc_neo = torch.zeros((B, Nlat, Nlon, self.total_sfc),device=x.device, dtype=x.dtype)
        xsfc_neo[:,:,:, :-self.n_const_vars] = xsfc
        xsfc_neo[:,:,:, -self.n_const_vars:] = self.const_data
        xsfc = xsfc_neo

        bad = 207.273, np.sqrt(5108.7202)
        good = 291.638, np.sqrt(105.98)
        sst_idx = 4+2#5*25 + 4 + 2
        if 'doctor' in self.config.name or 'ducati' in self.config.name:
            #print("prescaling SST", xsfc.shape)
            #prev = torch.nanmean(x_gpu[0][..., sst_idx].float())
            xsfc[..., 6] = xsfc[..., sst_idx] * good[1]/bad[1] + (good[0]-bad[0])/bad[1]

        assert xsfc.shape[-1] == self.total_sfc
        xsfc = xsfc.permute(0, 3, 1, 2)

        # nan to zero for sstk. input should be zero but target should have nans
        if torch.isnan(xsfc).any():
            # commented out for hres training
            #assert '034_sstk' in c.mesh.full_varlist, "sstk must be in full_varlist if there are NaNs"
            xsfc = torch.nan_to_num(xsfc, nan=0.0)

        # zeropadding should always be done ahead of time, not here
        # so, now the code just assrts that the inputs look right, and complains if they don't
        for i in range(c.mesh.n_sfc-c.mesh.extra_sfc_pad):
            assert torch.nonzero(xsfc[:,i]).any() or c.mesh.sfc_vars[i] == 'zeropad', f"xsfc[{i}] ({c.mesh.sfc_vars[i]}) is all zeros"

        # commented out for hres training
        #for i in range(c.mesh.n_sfc-c.mesh.extra_sfc_pad, c.mesh.n_sfc):
        #    assert not torch.nonzero(xsfc[:,i]).any(), f"xsfc[{i}] ({c.mesh.sfc_vars[i]}) is not all zeros"
        
        return xpr, xsfc
    
    @staticmethod
    def combine_pr_sfc(xpr_conv, xsfc_conv, c):
        xpr_conv = xpr_conv.permute(0, 2, 3, 4, 1)
        xsfc_conv = xsfc_conv.permute(0, 2, 3, 1)
        x_conv = torch.cat((xpr_conv, xsfc_conv[:, np.newaxis]), axis=1)
        x_conv = torch.flatten(x_conv, start_dim=1, end_dim=3)
        return x_conv
    
    def to(self, *args, **kwargs):
        global Gmatepoint_stream
        super().to(*args, **kwargs)
        #device = kwargs.get('device', args[0] if args else 'cuda' if torch.cuda.is_available() else 'cpu')
        Gmatepoint_stream = torch.cuda.Stream()

        if hasattr(self,'decoders'):
            try:
                for k,v in self.decoders.items():
                    v.to(*args, **kwargs)
            except:
                self.singledec.to(*args, **kwargs)
        return self
    
    def rollout(self, xs, time_horizon=None,dt_dict=None,min_dt=3, callback=None, stay_latent=False, dts=None):
        """
        Makes predictions going out to the time horizon in hours
        Does so at the max resolution the model can support
        For example, with a target time horizon of 48h, and support for 12h and 24h timestamps, it would generate 12, 24, 24 + 12, 24 + 24

        callback can be passed to get intermediate results, which can save memory

        dt_dict is a schedule of time resolutions, defined as start_forecast_hour: resolution
        For example, {6:3,24:6,72:12} would give min_dt until hour 6, 3 hour resolution until hour 24, etc

        dts lets you bypass dt_dict and other stuff and pass in a list of time steps to roll out manually
        """

        def all_to(xs, *args, **kwargs):
            return [x.to(*args, **kwargs) for x in xs]

        ts = self.config.timesteps

        if dts is None:
            if dt_dict is not None:
                dts = get_rollout_times(dt_dict,time_horizon=time_horizon,min_dt=min_dt)
                print(dts)
            else: 
                dts = np.arange(0,time_horizon+1,np.maximum(np.min(ts),min_dt))[1:]

        if stay_latent:
            assert self.config.parallel_encoders, "only implemented latent rollout for parallel encoders for now"
            xs = all_to(xs, 'cuda')
            t0s = xs[-1]
            n_encs = len(self.encoders)
            x_tr = None
            for i, encoder in enumerate(self.encoders):
                wt = 0.5
                if 'doctor' in self.config.name:
                    wt = 0.9 if i == 1 else 0.1
                x_e = encoder(xs[i], xs[2]) * wt
                if x_tr is None:
                    x_tr = x_e
                else:
                    x_tr += x_e
                del x_e
            
            x_initial = x_tr
            if self.config.decoder_reinput_initial:
                extra = self.downsample_latent(x_initial)

            assert not self.config.delta_processor, "delta processor not supported"
            
            dts = [0] + dts
            outputs = []
            for i, total_dt in enumerate(dts[:-1]):
                print(f"Rollout step {i+1}/{len(dts)-1}")
                ddt = dts[i+1] - dts[i]
                if isinstance(self.config.processor_dt, int):
                    assert ddt % self.config.processor_dt == 0
                    for _ in range(ddt//self.config.processor_dt):
                        x_tr = self.proc_swin(x_tr)
                else:
                    possible_dts = [k for k in self.config.processor_dt if ddt % k == 0]
                    max_dt = max(possible_dts)
                    for _ in range(ddt//max_dt):
                        print(f"ddt={ddt}, using processor[{max_dt}]")
                        x_tr = self.processors[str(max_dt)](x_tr)
                if self.config.decoder_reinput_initial:
                    partial = torch.cat([x_tr, extra], dim=-1)
                else:
                    partial = x_tr
                y = self.singledec(partial)
                y = set_metadata(y, t0s + 3600*total_dt)
                if callback:
                    callback(dts[i], y)
                else:
                    outputs.append(y)
            if not callback: return outputs
        else:
            progress = [
                SimpleNamespace(
                    target_dt=dtt,
                    tensors=xs,
                    accum_dt=0,
                    steps_left=min_additions(ts,dtt)
                ) for dtt in dts
            ]
            finished = {}
            while progress:
                min_t = np.min([v.accum_dt for v in progress])
                sub = [v for v in progress if v.accum_dt == min_t]
                step = sub[0].steps_left[0]
                sub = [v for v in sub if v.steps_left[0] == step]
                print(f"STEP step_from: {sub[0].accum_dt} step: {step} progress={len(progress)}")
                new_tensors = [self.forward(all_to(sub[0].tensors,'cuda'),dt=step,gimme_deltas=False), sub[0].tensors[-1] + step * 3600]
                #print(new_tensors[0].meta)
                #print(type(new_tensors[0]))
                new_tensors = all_to(new_tensors,'cpu')
                #print(new_tensors[0].meta)
                check_dt = sub[0].accum_dt
                for v in sub:
                    assert v.accum_dt == check_dt
                    v.tensors = new_tensors
                    v.accum_dt += step
                    v.steps_left = v.steps_left[1:]
                    if not v.steps_left:
                        progress.remove(v)
                        assert v.target_dt not in finished
                        output_dt_s = v.tensors[-1] - xs[-1].to('cpu')
                        assert abs(output_dt_s - v.target_dt*3600) < 2*60, f'Output and target dt hours do not match: {round(output_dt_s.item()/3600, 2)} vs {v.target_dt}'
                        if callback:
                            callback(v.target_dt, v.tensors[0])
                        else:
                            finished[v.target_dt] = v.tensors[0]

            return finished




def xonly(xs):
    assert len(xs) == 2, f"len(xs) {len(xs)}, this needs to be 1 right now. Not yet are we taking multiple sources"
    x = xs[0]
    t0 = int(xs[1].item())
    return x,t0

class ForecastStepDecoder(ForecastStepBase):
    def __init__(self,dt=24,config=None):
        super().__init__(config)
        c = self.config
        self.dt = dt

        H = c.hidden_dim
        if c.decoder_reinput_initial:
            H += c.decoder_reinput_size
        
        num_heads = H // c.dims_per_head

        if c.dec_swin_depth > 0:
            self.dec_swin = c.Transformer(config = self.config, dim=H, padded_lon=c.padded_lon, input_resolution=c.resolution, depth=c.dec_swin_depth, num_heads=num_heads, window_size=c.window_size, FLASH=c.FLASH, drop_path=c.drop_path)
        else: 
            self.dec_swin = nn.Identity()

        
        self.deconv = nn.ConvTranspose3d(out_channels=c.output_mesh.n_pr_vars, in_channels=H, kernel_size=c.patch_size, stride=c.patch_size)
        self.deconv_sfc = nn.ConvTranspose2d(out_channels=c.output_mesh.n_sfc_vars, in_channels=H, kernel_size=c.patch_size[1:], stride=c.patch_size[1:])

        if c.output_deltas:
            delta_norm_matrix = torch.from_numpy(load_delta_norm(dt, c.output_mesh.n_levels, c.output_mesh)[1])
            self.register_buffer("delta_norm_matrix", delta_norm_matrix)

        self.H = H            
    
    def forward(self, x):
        B = x.shape[0]
        c = self.config
        x_tr = self.dec_swin(x)

        x_tr = x_tr.permute(0, 2, 1)
        x_tr = x_tr.view(B, self.H, c.resolution[0], c.resolution[1], c.resolution[2])
        if c.padded_lon:
            pad = c.window_size[2]//2
            x_tr = x_tr[:, :, :, :, pad:-pad]

        y_pr_conv = x_tr[:, :, :-1, :, :]
        y_sfc_conv = x_tr[:, :, -1, :, :]

        if c.checkpoint_convs:
            y_pr = call_checkpointed(self,self.deconv, y_pr_conv)
            y_sfc = call_checkpointed(self,self.deconv_sfc, y_sfc_conv)
        else:
            y_pr = self.deconv(y_pr_conv)
            y_sfc = self.deconv_sfc(y_sfc_conv)

        y_sfc = y_sfc.permute(0, 2, 3, 1)
        bad = 207.273, np.sqrt(5108.7202)
        good = 291.638, np.sqrt(105.98)
        sst_idx = 4+2#5*25 + 4 + 2
        if 'doctor' in self.config.name or 'ducati' in self.config.name:
            #print("postscaling SST")
            y_sfc[..., sst_idx] = y_sfc[..., sst_idx] * bad[1]/good[1] + (bad[0]-good[0])/good[1]


        y_pr = y_pr.permute(0, 3, 4, 1, 2)
        y_pr = torch.flatten(y_pr, start_dim=-2)
        y = torch.cat((y_pr, y_sfc), axis=-1)
        return y



    
def pad_lon(x, pad):
    B, D, H, W, C = x.shape
    full = torch.zeros((B, D, H, W + 2*pad, C), dtype=x.dtype, device=x.device)
    full[:, :, :, pad:pad+W, :] = x
    full[:, :, :, :pad, :] = x[:, :, :, W-pad:W]
    full[:, :, :, pad+W:, :] = x[:, :, :, pad:pad+pad]
    return full

class ForecastStepUDecoder(ForecastStepBase):
    def __init__(self, dt=0, config=None):
        super().__init__(config)
        cm = self.config
        c = self.config.ucodec_config
        assert c.padded_lon, "padded_lon must be true"

        self.winpad = c.tr_win[2] // 2 * 2


        ks_sfc = c.conv_sz[0][1:]
        ks_pr = c.conv_sz[0]
        if c.blend_conv:
            ks_sfc = [x*2 for x in ks_sfc]
            ks_pr = [ks_pr[0]]+ [x*2 for x in ks_pr[1:]]

        self.deconv_sfc = nn.ConvTranspose2d(
            out_channels=cm.output_mesh.n_sfc_vars,
            in_channels=c.conv_dim[0]*2,
            kernel_size=ks_sfc,
            stride=c.conv_sz[0][1:]
        )
        self.deconv_pr = nn.ConvTranspose3d(
            out_channels=cm.output_mesh.n_pr_vars,
            in_channels=c.conv_dim[0],
            kernel_size=ks_pr,
            stride=c.conv_sz[0]
        )
        
        self.conv_addl_sfc = nn.Conv2d(
            in_channels=self.n_const_vars,
            out_channels=c.conv_dim[0]*2,
            kernel_size=c.conv_sz[0][1:],
            stride=c.conv_sz[0][1:]
        )

        self.const_data = self.const_data.unsqueeze(0).permute(0,3,1,2)

        self.deconvs = nn.ModuleList([
            nn.ConvTranspose3d(
                out_channels=c.conv_dim[i],
                in_channels=c.conv_dim[i+1],
                kernel_size=c.conv_sz[i+1],
                stride=c.conv_sz[i+1]
            )
            for i in range(len(c.conv_sz) - 1)
        ])

        res = (cm.output_mesh.n_levels // c.conv_sz[0][0] + 2, cm.output_mesh.lats.size // c.conv_sz[0][1], cm.output_mesh.lons.size // c.conv_sz[0][2] + self.winpad)
        self.transformers = nn.ModuleList()
        for i in range(len(c.conv_sz)):
            if i > 0:
                res = (res[0] // c.conv_sz[i][0], res[1] // c.conv_sz[i][1], (res[2] - self.winpad) // c.conv_sz[i][2] + self.winpad)
            self.transformers.append(
                self.config.Transformer(
                    dim=c.conv_dim[i],
                    depth=c.tr_depth[i],
                    num_heads=c.conv_dim[i] // c.tr_headdim[i],
                    window_size=c.tr_win,
                    input_resolution=res,
                    config=cm,
                    padded_lon=c.padded_lon,
                    checkpoint_chunks = c.tr_checkpoint_chunks[i],
                    FLASH=self.config.FLASH
                )
            )

        assert c.conv_dim[-1] == self.config.hidden_dim, f"Last conv_dim {c.conv_dim[-1]} != hidden_dim {self.config.hidden_dim}"
        assert res[:-1] == self.config.resolution[:-1], f"Last resolution {res} != config resolution {self.config.resolution}"
        assert res[2] - self.winpad == self.config.resolution[2] - self.config.window_size[2]//2*2, f"Last resolution {res} != config resolution {self.config.resolution}"

    def forward(self, x):
        c = self.config
        B = x.shape[0]
        C = x.shape[-1]

        winpad = self.winpad // 2
        import_pad = c.window_size[2]//2 - winpad

        D, H, W = self.transformers[-1].input_resolution
        W += 2*import_pad
        x = x.view(B, D, H, W, C)[:,:,:,import_pad:-import_pad,:]
        W -= 2*import_pad
        x = x.reshape(B, D*H*W, C)

        for i in range(len(self.transformers) - 1, 0, -1):
            x = self.transformers[i](x)
            x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
            x = x[:, :, :, :, winpad:-winpad]
            x = self.deconvs[i-1](x)
            x = pad_lon(x.permute(0, 2, 3, 4, 1), winpad)
            B, D, H, W, C = x.shape
            x = x.view(B, D*H*W, C)
            if c.sincos:
                pe = posemb_sincos_3d(((0, D, H, W, C), x.device, x.dtype))
                x = x + pe[None]
        
        x.view(B,D,H,W,C)[:,-2:,:,:,:] += pad_lon(self.conv_addl_sfc(self.const_data).view(B,C,2,H,W-2*winpad).permute(0,2,3,4,1),winpad)
        x = self.transformers[0](x)
        x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        x = x[:, :, :, :, winpad:-winpad]

        W = W - 2*winpad
        y_pr_conv = x[:, :, :-2, :, :]
        y_sfc_conv = x[:, :, -2:, :, :]
        
        y_sfc_conv = y_sfc_conv.reshape(B, C*2, H, W)

        y_pr = self.deconv_pr(y_pr_conv)
        y_sfc = self.deconv_sfc(y_sfc_conv) 


        if c.ucodec_config.blend_conv:
            _,tr1,tr2 = [x // 4 for x in self.deconv_pr.kernel_size]
            y_pr[:,:,:,:,tr2:tr2*2]+=y_pr[:,:,:,:,-tr2:]  # wrap around right side to the left side
            y_pr[:,:,:,:,-tr2*2:-tr2]+=y_pr[:,:,:,:,:tr2] # wrap around left side to the right side
            y_pr = y_pr[:,:,:,tr1:-tr1,tr2:-tr2] # cut off the padding on the sides


            _,tr1 = [x // 4 for x in self.deconv_sfc.kernel_size]
            y_sfc[:,:,:,tr1:tr1*2] += y_sfc[:,:,:,-tr1:]  # wrap around right side to the left side
            y_sfc[:,:,:,-tr1*2:-tr1] += y_sfc[:,:,:,:tr1] # wrap around left side to the right side
            y_sfc = y_sfc[:,:,tr1:-tr1,tr1:-tr1] # cut off the padding on the sides
            
            # FUCK THE POLES

        y_pr = y_pr.permute(0, 3, 4, 1, 2)
        y_sfc = y_sfc.permute(0, 2, 3, 1)
        y_pr = torch.flatten(y_pr, start_dim=-2)
        y = torch.cat((y_pr, y_sfc), axis=-1)
        return y


class UCodecConfig:
    def __init__(self, **kwargs):
        self.conv_sz = [(2,2,2), (1,2,2), (2,2,2), (1,2,2)]
        self.conv_dim = [48, 128, 512, 1280]
        self.tr_headdim = [16, 16, 32, 32]
        self.tr_checkpoint_chunks = [1,1,1,1]
        self.tr_win = (3,3,3)
        self.tr_depth = [2, 2, 2, 4]
        self.padded_lon = True
        self.sincos = True
        self.blend_conv = False

        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a UCodecConfig attribute"
            setattr(self,k,v)
        self.update()

    def update(self):
        assert len(self.conv_sz) == len(self.conv_dim) == len(self.tr_headdim) == len(self.tr_depth)




class ForecastStepUEncoder(ForecastStepBase):
    def __init__(self, config=None):
        super().__init__(config)
        cm = self.config
        c = self.config.ucodec_config
        assert c.padded_lon, "padded_lon must be true"

        self.winpad = c.tr_win[2] // 2 * 2

        self.conv_sfc = nn.Conv2d(
            in_channels=self.total_sfc,
            out_channels=c.conv_dim[0]*2,
            kernel_size=c.conv_sz[0][1:],
            stride=c.conv_sz[0][1:]
        )
        self.conv_pr = nn.Conv3d(
            in_channels=cm.mesh.n_pr_vars,
            out_channels=c.conv_dim[0],
            kernel_size=c.conv_sz[0],
            stride=c.conv_sz[0]
        )
        
        self.convs = nn.ModuleList([
            nn.Conv3d(
                in_channels=c.conv_dim[i],
                out_channels=c.conv_dim[i+1],
                kernel_size=c.conv_sz[i+1],
                stride=c.conv_sz[i+1]
            )
            for i in range(len(c.conv_sz) - 1)
        ])

        res = (cm.mesh.n_levels // c.conv_sz[0][0] + 2, cm.mesh.lats.size // c.conv_sz[0][1], cm.mesh.lons.size // c.conv_sz[0][2] + self.winpad)
        self.transformers = nn.ModuleList()
        for i in range(len(c.conv_sz)):
            if i > 0:
                res = (res[0] // c.conv_sz[i][0], res[1] // c.conv_sz[i][1], (res[2] - self.winpad) // c.conv_sz[i][2] + self.winpad)
            self.transformers.append(
                self.config.Transformer(
                    dim=c.conv_dim[i],
                    depth=c.tr_depth[i],
                    num_heads=c.conv_dim[i] // c.tr_headdim[i],
                    window_size=c.tr_win,
                    input_resolution=res,
                    config=cm,
                    padded_lon=c.padded_lon,
                    checkpoint_chunks = c.tr_checkpoint_chunks[i],
                    FLASH=self.config.FLASH
                )
            )

        assert c.conv_dim[-1] == self.config.hidden_dim, f"Last conv_dim {c.conv_dim[-1]} != hidden_dim {self.config.hidden_dim}"
        assert res[:-1] == self.config.resolution[:-1], f"Last resolution {res} != config resolution {self.config.resolution}"
        assert res[2] - self.winpad == self.config.resolution[2] - self.config.window_size[2]//2*2, f"Last resolution {res} != config resolution {self.config.resolution}"

    def forward(self, x, t0s):
        c = self.config.ucodec_config
        cm = self.config

        xpr, xsfc = self.breakup_pr_sfc(x, t0s)

        xpr_conv = self.conv_pr(xpr)
        xsfc_conv = self.conv_sfc(xsfc)
        xsfc_conv = xsfc_conv.view([xsfc_conv.shape[0], xsfc_conv.shape[1] // 2, 2] + list(xsfc_conv.shape[2:]))

        xpr_conv = xpr_conv.permute(0, 2, 3, 4, 1)
        xsfc_conv = xsfc_conv.permute(0, 2, 3, 4, 1)
        x = torch.cat((xpr_conv, xsfc_conv), axis=1)

        pad = self.winpad // 2
        x = pad_lon(x, pad)

        B, D, H, W, C = x.shape

        for i in range(len(c.conv_sz)):
            x = x.view(B, D*H*W, C)

            if c.sincos:
                pe = posemb_sincos_3d(((0, D, H, W, C), x.device, x.dtype))
                x = x + pe[None]

            x = self.transformers[i](x)
            x = x.view(B, D, H, W, C)
            if i == len(c.conv_sz)-1:
                break
            x = x.permute(0, 4, 1, 2, 3)
            x = x[:, :, :, :, pad:-pad]
            x = self.convs[i](x)
            x = x.permute(0, 2, 3, 4, 1)
            x = pad_lon(x, pad)
            B, D, H, W, C = x.shape

        export_pad = cm.window_size[2]//2 - pad
        x = pad_lon(x, export_pad)
        W = W + export_pad*2
        x = x.view(B, D*H*W, C)

        assert C == cm.hidden_dim, f"C {C} c.hidden_dim {cm.hidden_dim}"
        assert D == cm.resolution[0] and H == cm.resolution[1] and W == cm.resolution[2], f"(D,H,W) ({D},{H},{W}) c.resolution {cm.resolution}"

        return x



class ForecastStepEncoder(ForecastStepBase):
    def __init__(self,config=None):
        super().__init__(config)
        c = self.config

        self.conv = nn.Conv3d(in_channels=self.mesh.n_pr_vars, out_channels=c.conv_dim, kernel_size=c.patch_size, stride=c.patch_size)
        self.conv_sfc = nn.Conv2d(in_channels=self.total_sfc, out_channels=c.conv_dim, kernel_size=c.patch_size[1:], stride=c.patch_size[1:])

        if c.enc_swin_depth > 0:
            self.enc_swin = c.Transformer(config=c, dim=c.hidden_dim, padded_lon=c.padded_lon, input_resolution=c.resolution, depth=c.enc_swin_depth, num_heads=c.num_heads, window_size=c.window_size, FLASH=c.FLASH, drop_path=c.drop_path)
        else:
            self.enc_swin = nn.Identity()

    
    def forward(self, x, t0s):
        c = self.config
        x_conv = self.conv_forward(x,t0s)

        if c.padded_lon:
            # longitude is the second to last dimension. pad it with zeros on c.window_size[2]//2 on either side
            full = torch.zeros((x_conv.shape[0], *c.resolution, c.hidden_dim), dtype=x_conv.dtype, device=x_conv.device)
            pad = (c.window_size[2]//2)
            w = c.resolution[2] - 2*pad
            x_convd = x_conv.view(x_conv.shape[0], c.resolution[0], c.resolution[1], c.resolution[2] - 2*pad, c.hidden_dim)
            full[:, :, :, pad:pad+w, :] = x_convd
            full[:, :, :, :pad, :] = x_convd[:, :, :, w-pad:w]
            full[:, :, :, pad+w:, :] = x_convd[:, :, :, pad:pad+pad]
            x_conv = full.view(x_conv.shape[0], np.prod(c.resolution), c.hidden_dim)

        if c.sincos:
            pe = posemb_sincos_3d(((0, c.resolution[0], c.resolution[1], c.resolution[2], c.hidden_dim), x_conv.device, x_conv.dtype))
            x_conv = x_conv + pe[None]

        x_tr = self.enc_swin(x_conv)

        return x_tr
    
    def conv_forward(self, x, t0s):
        c = self.config
        xpr,xsfc = self.breakup_pr_sfc(x, t0s)

        xpr_conv = self.conv(xpr)
        xsfc_conv = self.conv_sfc(xsfc)
 
        x_conv = ForecastStepBase.combine_pr_sfc(xpr_conv, xsfc_conv, self.config)

        return x_conv


class OverengineeredAllocTable:
    def __init__(self, sizes):
        self.arrays = {}
        tsz = 0
        tlg = 0
        assert len(sizes) == 1, "dual stuff not implemented yet"
        for sz in sizes:
            print("making bigtable! size:", sz)
            total = np.prod(sz) * 2
            lg = np.log2(total)
            eff = total/(2**np.ceil(lg))
            arrs = []
            done = 0
            n = 0
            while done < sz[0] and n < 20:
                remaining = sz[0] - done
                total = np.prod((remaining, sz[1], sz[2]))*2
                lg = np.log2(total)
                doable = int(np.floor((2**np.floor(lg))/(np.prod(sz[1:])*2)))
                if doable == 0: doable = 1
                szp = (doable, sz[1], sz[2])
                ss = np.prod(szp)*2
                tsz += ss 
                tlg += 2**np.ceil(np.log2(ss))
                print("doing", szp)
                arrs.append(torch.empty(szp, dtype=torch.float16, pin_memory=True))
                done += doable
                #under = np.floor(lg)
                #new = sz[0] * 2**(under-lg)
                #print("doing", doable, "instead of", sz)
                n += 1
            assert n < 18, "uhhh sth cursed with this algorithm lol"
            self.arrays[(sz[1], sz[2])] = arrs
        print("Overall efficiency %.2f%%" % (100 * tsz/tlg))

    """
    def copy(self, idx, data):
        sh = data.shape
        assert sh in self.arr
        self[idx].copy_(
    """


    def __getitem__(self, idx):
        for arr in self.arrays[list(self.arrays.keys())[0]]:
            l = len(arr)
            if idx < l:
                return arr[idx]
            idx -= l
        assert False


class ForecastStep3D(ForecastStepBase):

    @TIMEIT()
    def __init__(self, config):
        self.config = config
        c = self.config
        super().__init__(config)
        self.code_gen = 'gen2'
        self.last_steps = None
        self.matepoint_ctx = []
        c.matepoint_ctx = self.matepoint_ctx # hacky af but yolo

        if not c.parallel_encoders:
            self.encoder = c.Encoder(config=config)
        else:
            assert len(c.inputs) > 1, "you don't need parallel encoders with only one input"
            self.encoders = nn.ModuleList()
            for _ in range(len(c.inputs)):
                encoder = c.Encoder(config=config)
                self.encoders.append(encoder)

        if c.rollout_reencoder:
            self.rollout_reencoder = ForecastStepEncoder(config=config)
        

        self.processors = nn.ModuleDict()
        for i,dt in enumerate(c.processor_dt):
            self.processors[str(dt)] = c.Transformer(config=c,dim=c.hidden_dim, padded_lon=c.padded_lon, input_resolution=c.resolution, depth=c.proc_depths[i], num_heads=c.num_heads, window_size=c.window_size, FLASH=c.FLASH, drop_path=c.drop_path, perturber=c.perturber)

        if not c.output_deltas:
            self.singledec = c.Decoder(dt=0, config=self.config)
            class ihmlawtd:
                def __init__(lol): pass
                def __getitem__(lol, key): return self.singledec
            self.decoders = ihmlawtd()
        else:
            assert False
            self.decoders = nn.ModuleDict()
            for t in c.timesteps:
                self.decoders[str(t)] = ForecastStepDecoder(dt=t, config=self.config)

        if c.decoder_reinput_initial:
            self.downsample_latent = nn.Linear(c.conv_dim, c.decoder_reinput_size)

    def forward_inner(self, xs, todo_dict, send_to_cpu, callback):
        if self.config.matepoint_table not in [None, False, 0] and matepoint.bigTable is None:
            tb = self.config.matepoint_table
            if tb == 'auto':
                print('automatically picking matepoint table based on number of timesteps')
                assert len(self.config.processor_dt) == 1, "automatic picker is not smart"
                assert self.config.Encoder != ForecastStepUEncoder, "automatic picker really is not smart"
                tb = max(todo_dict.keys()) * self.config.proc_swin_depth//self.config.processor_dt[0] + len(todo_dict.keys())*self.config.dec_swin_depth + self.config.enc_swin_depth
            res = np.prod(self.config.resolution)
            matepoint.bigTable = OverengineeredAllocTable([(tb, res, self.config.hidden_dim)])
        matepoint.matepoint_ctx = []
        self.config.matepoint_ctx = []

        matepoint.bigN = 0

        if self.name:
            if 'bachelor' in self.name or 'master' in self.name or 'doctorate'  in self.name:
                assert self.const_vars == ['lat','lon','land_mask','soil_type','elevation']
                assert self.n_const_vars == 12
        
        # Note: In future, it might be necessary to move finished or intermediate steps outputs back to the cpu in order to save vram

        if hasattr(self, 'output_deltas') and hasattr(self.config, 'output_deltas'):
            assert self.output_deltas == self.config.output_deltas, "Ohp, Joan I changed output deltas to be in the config"
        c = self.config
        assert not c.delta_processor, "delta processor not supported"
        assert not c.return_random, "return random not supported"

        t0s = xs[-1]
        xs = xs[0:-1]
        
        if len(xs) != len(c.inputs):
            for dt, todo in todo_dict.items():
                assert todo.startswith('rE'), f"number of input {len(xs)} does not match the expected number of input {len(c.inputs)}. have to use rollout reencoder"

        def encode(xs,t0s):
            if c.parallel_encoders:
                num_encs = len(self.encoders)
                assert len(xs) == num_encs, f"number of inputs does not match number of encoders. Probably should be calling the rollout reencoder. {len(xs)} vs {num_encs}"
                x = None
                for i,encoder in enumerate(self.encoders):
                    wt = 0.5
                    if 'doctor' in self.config.name:
                        wt = 0.9 if i == 1 else 0.1
                    x_e = encoder(xs[i],t0s) * wt
                    if x == None: x = x_e
                    else: x = x + x_e
                    del x_e
            else:
                x = self.encoder(xs[0],t0s) 
            return x
        
        todos = [SimpleNamespace(
                target_dt=k,
                remaining_steps = v.split(','),
                completed_steps = [],
                state = (xs,None),
                accum_dt=0,
            ) for k,v in todo_dict.items()]

        outputs = {}
        while todos:
            tnow = todos[0]
            step = tnow.remaining_steps[0]
            completed_steps = tnow.completed_steps.copy()
            x,extra = tnow.state
            accumulated_dt = 0
            #for todo in todos:
            #    print(f"{todo.target_dt}: {','.join(todo.remaining_steps)}")
            if step == 'E':
                x = encode([xx.clone() for xx in x],t0s+3600*tnow.accum_dt)
                if c.decoder_reinput_initial:
                    extra = self.downsample_latent(x.clone())
            elif step == 'rE':
                assert c.rollout_reencoder, "rollout reencoder not enabled"
                x = self.rollout_reencoder(x[0],t0s+3600*tnow.accum_dt)
                if c.decoder_reinput_initial:
                    extra = self.downsample_latent(x)
            elif step.startswith('P'):
                pdt = int(step[1:])
                x = self.processors[str(pdt)](x)
                accumulated_dt += pdt
            elif step == 'D':
                if c.decoder_reinput_initial:
                    x = torch.cat((x, extra.clone()), dim=-1)
                decoder = self.decoders[0]
                x = [decoder(x)]
                if send_to_cpu:
                    x = [xx.cpu() for xx in x]
            else:
                assert False, f"Unknown step {step}"
            for todo in todos:
                if todo.remaining_steps[0] == step and todo.completed_steps == completed_steps:
                    todo.state = (x,extra)
                    todo.accum_dt += accumulated_dt
                    todo.remaining_steps = todo.remaining_steps[1:]
                    todo.completed_steps.append(step)
                    if not todo.remaining_steps:
                        assert len(x) == 1
                        if callback is not None:
                            callback(todo.target_dt, x[0])
                        else:
                            outputs[todo.target_dt] = x[0]
                        todos.remove(todo)
        return outputs
    
    @staticmethod
    def simple_gen_todo(dts,processor_dts):
        out = {}
        for dt in dts:
            rem_dt = dt
            todo = "E,"
            for pdt in reversed(processor_dts):
                num = rem_dt // pdt
                rem_dt -= pdt*num
                todo+= f"P{pdt},"*num
                if rem_dt == 0:
                    break
            assert rem_dt == 0
            todo+="D"
            out[dt] = todo
        return out
 
    def forward(self, xs, todo, send_to_cpu=False, callback=None):
        def is_list_of_ints(obj): return isinstance(obj, list) and all(isinstance(x, int) for x in obj)
        if is_list_of_ints(todo):
            if 'doctor' in self.name or 'ducati' in self.name:
                todo = self.simple_gen_todo(sorted(todo),[1,6])
                for x in todo:
                    todo[x] = todo[x].replace('P6', 'P3,P3')
            else:
                todo = self.simple_gen_todo(sorted(todo),self.config.processor_dt)
        y = self.forward_inner(xs, todo, send_to_cpu, callback)
        return y
    
    def change_droppath_prob(self, prob):
        for name, module in self.named_modules():
            if isinstance(module, DropPath):
                module.drop_prob = prob



class ForecastStepDiffusion(nn.Module,SourceCodeLogger):
    def __init__(self, forecaster, diffuser, T=1000):
        super().__init__()
        self.config = forecaster.config
        self.mesh = forecaster.config.output_mesh
        self.forecaster = forecaster
        self.diffuser = diffuser


        self.T = T
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.T)  # (T,)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # (T,)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)


        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)



    def predict(self, x_full, dT):
        pdT = self.forecaster.config.processor_dt[0]
        with torch.no_grad():
            assert self.forecaster.mesh.extra_sfc_pad == 3
            x_clone = [xx.clone() for xx in x_full]
            for i in range(3):
                assert not (x_clone[0][:,:,:,-3+i] == 0).all() 
                x_clone[0][:,:,:,-3+i] = 0 # need to zero out for bachelor 
            todo = {dT : ','.join(['E'] + [f'P{pdT}']*(dT//pdT))}
            c = self.forecaster(x_clone, todo)[dT]
            c = c.view(*self.forecaster.config.resolution, self.forecaster.config.hidden_dim) # D H W C
            
            assert self.forecaster.config.padded_lon
            assert (self.forecaster.config.window_size[2]-1) / 2 == 3.0
            c = c[-1,:,3:-3] # -1 is for surface 
            c = c.permute(2,0,1).unsqueeze(0)
        return c
        


    def forward(self, x_full, dT, y_t, t):
        # little t is time in diffusion, big T is time in forecast
        
        # x_t: the sample with the noise added
        # t: the timestep for the diffusion process
        # x_full: the full weather instance for the predictive model
        # dT: forecast hour

        c = self.predict(x_full, dT)
            
        noise_pred = self.diffuser(y_t, t, c) 

        return noise_pred

    def generate_slow(self, x_full, dT):
        c = self.predict(x_full, dT)
        device = c.device

        B = 1
        sample = torch.randn(B, self.mesh.n_sfc_vars, len(self.mesh.lats), len(self.mesh.lons), device=device)
        for i in reversed(range(self.T)):
            print("Generating Step:",i)
            t_batch = torch.full((B,), i, device=device, dtype=torch.long)
            noise_pred = self.diffuser(sample, t_batch, c)

            if i > 0:
                noise = torch.randn_like(sample) if i > 1 else torch.zeros_like(sample)
                sample = (1 / torch.sqrt(self.alphas[i])) * (sample - ((self.betas[i] / self.sqrt_one_minus_alphas_cumprod[i]) * noise_pred)) + torch.sqrt(self.betas[i] * (1.0 - self.alphas_cumprod[i - 1]) / (1.0 - self.alphas_cumprod[i])) * noise
            else:
                sample = (1 / torch.sqrt(self.alphas[0])) * (sample - ((self.betas[0] / self.sqrt_one_minus_alphas_cumprod[0]) * noise_pred))
        sample.permute(0,2,3,1)

        #edges are fucked rn
        sample[:,:5,:,:] = 0
        sample[:,-5:,:] = 0
        sample[:,:,:5,:] = 0
        sample[:,:,-5:,:] = 0

        return sample
        #generated = torch.clamp(sample, -1.0, 1.0)
    
    def generate(self, x_full, dT, steps=None):
        c = self.predict(x_full, dT)
        device = c.device

        B = 1
        sample = torch.randn(B, self.mesh.n_sfc_vars, len(self.mesh.lats), len(self.mesh.lons), device=device)
        
        # If num_steps is not provided, use the default number of steps
        num_steps = steps or self.T
        
        # Create a new time step sequence
        time_steps = torch.linspace(self.T - 1, 0, num_steps, device=device)
        
        for i in range(num_steps):
            print(f"Generating Step: {i+1}/{num_steps}")
            t = time_steps[i]
            t_prev = time_steps[i+1] if i < num_steps - 1 else torch.tensor(-1, device=device)
            
            # Interpolate alpha values
            alpha = self.alphas_cumprod[t.long()]
            alpha_prev = self.alphas_cumprod[t_prev.long()] if i < num_steps - 1 else torch.tensor(1.0, device=device)
            
            t_batch = t.repeat(B)
            noise_pred = self.diffuser(sample, t_batch, c)
            
            # Compute the variance
            variance = (1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)
            
            # Compute the "direction pointing to x_t"
            pred_original_sample = (sample - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
            
            # Compute x_{t-1}
            sample = torch.sqrt(alpha_prev) * pred_original_sample + \
                    torch.sqrt(1 - alpha_prev - variance) * noise_pred + \
                    torch.sqrt(variance) * torch.randn_like(sample)

        sample = sample.permute(0,2,3,1)
        print(sample.shape)
        #edges are fucked rn
        sample[:, :5,:,:] = 0; 
        sample[:,-5:,:,:] = 0; 
        sample[:,:, :5,:] = 0; 
        sample[:,:,-5:,:] = 0

        return sample
    

from model_latlon.primatives2d import southpole_pad2d, southpole_unpad2d
from model_latlon.codec2d import HarebrainedAutoEncoder2d

class SfcOnlyAutoencoder(nn.Module,SourceCodeLogger):
    def __init__(self,mesh):
        super().__init__()
        self.mesh = mesh
        self.inner_model = HarebrainedAutoEncoder2d(mesh)

    def get_matepoint_tensors(self):
        return []

    def forward(self, x, dts):
        assert dts == [0]
        xs = x[0][:,:,:,-self.mesh.n_sfc_vars:]
        B,H,W,C = xs.shape
        xs = xs.permute(0,3,1,2)
        ys = self.inner_model(xs)
        ys = ys.permute(0,2,3,1)
        out = torch.zeros_like(x[0])
        out[...,:-self.mesh.n_sfc_vars] = x[0][...,:-self.mesh.n_sfc_vars]
        out[...,-self.mesh.n_sfc_vars:] = ys
        return {0:out}
    
class SfcOnlyHareEncoder(nn.Module,SourceCodeLogger):
    def __init__(self,mesh):
        super().__init__()
        self.mesh = mesh

        self.down = HarebrainedConvDown2d(self.mesh.n_sfc_vars, 16)
        self.conv = HarebrainedConv2d(16,16)
        self.up = HarebrainedConvUp2d(16, self.mesh.n_sfc_vars)


    def get_matepoint_tensors(self):
        return []

    def forward(self, x, dts):
        assert dts == [0]
        xs = x[0][:,:,:,-self.mesh.n_sfc_vars:]
        B,H,W,C = xs.shape
        xs = xs.permute(0,3,1,2)
        xs = southpole_pad2d(xs)
        xs = self.down(xs)
        xs = self.conv(xs)
        ys = self.up(xs)
        ys = southpole_unpad2d(ys)
        ys = ys.permute(0,2,3,1)
        out = torch.zeros_like(x[0])
        out[...,:-self.mesh.n_sfc_vars] = x[0][...,:-self.mesh.n_sfc_vars]
        out[...,-self.mesh.n_sfc_vars:] = ys
        return {0:out}



