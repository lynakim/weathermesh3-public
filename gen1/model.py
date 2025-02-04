import copy
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from types import SimpleNamespace

from gen1.utils import (
    CONSTS_PATH,
    levels_medium,
    SourceCodeLogger,
    get_date,
    dimprint,
    sizeprint,
    print_mem,
    get_rollout_times,
    min_additions,
    load_delta_norm,
    set_metadata,
    to_mesh,
)
from gen1.data import EarthSpecificModel
import gen1.matepoint as matepoint
from gen1.matepoint import matepoint_pipeline

Gmatepoint_stream = None
class ForecastStepConfig():
    def __init__(self,inputs,**kwargs):
        self.inputs = inputs
        self.outputs = inputs
        self.patch_size = (4,8,8)
        self.preconv_hidden_dim = 128
        self.hidden_dim = 768
        self.FLASH = True
        self.dims_per_head = 32
        self.drop_path = 0.0
        self.checkpoint_every = 1
        self.skip_every = None
        self.window_size = (2,6,12)
        self.lat_compress = False
        self.delta_processor = False
        self.activation = nn.GELU()

        self.adapter_swin_depth = 0
        self.enc_swin_depth = 0
        self.proc_swin_depth = 24
        self.dec_swin_depth = 0
        self.timesteps = [24]
        self.train_timesteps = None
        self.processor_dt = None
        self.return_random = False

        self.surface_sandwich_bro = False
        self.output_deltas = True
        self.use_matepoint = False
        self.checkpointfn = None#matepoint.checkpoint

        self.decoder_reinput_initial = False
        self.decoder_reinput_size = 64

        self.perturber = 0

        self.adapter_H1 = 16
        self.adapter_dim_seq = [64,128,256]

        self.adapter_use_input_bias = not True

        self.neorad = False
        self.neorad_subsamp = True
            
        self.__dict__.update(kwargs)
        self.update()

    def update(self): 
        if self.train_timesteps is None:
            self.train_timesteps = self.timesteps
        self.mesh = self.inputs[0]
        self.output_mesh = self.outputs[0]

        if self.processor_dt is None:
            self.processor_dt = self.timesteps[0]
        for t in self.timesteps:
            if self.processor_dt == 0: assert t == 0, "timesteps must be 0 if processor_dt is 0"
            else: assert t % self.processor_dt  == 0, f"processor_dt {self.processor_dt} must divide all timesteps {self.timesteps}"

        assert self.hidden_dim % self.dims_per_head == 0
        self.num_heads = self.hidden_dim // self.dims_per_head

        self.resolution = ((self.mesh.n_levels+self.surface_sandwich_bro)//self.patch_size[0] + 1, self.mesh.lats.shape[0] // self.patch_size[1], 
            self.mesh.lons.shape[0] // self.patch_size[2]) 
 
        #self.resolution = (self.mesh.n_levels//self.patch_size[0] + 1 + 1, self.mesh.lats.shape[0] // self.patch_size[1], 
        #    self.mesh.lons.shape[0] // self.patch_size[2])  #XXX HERE!!
        
        if self.lat_compress:
            n1 = self.mesh.lats.shape[0] // self.patch_size[1]
            w1 = self.window_size[1]
            self.post_conv_nlat = n1
            self.lat_segment_size = (n1 - ((n1 - n1//8*2)//w1 * w1)) // 2 #lat segment size is essentially 1/8 of the lat size (post conv), but with bs to make the total divisible by window size
            self.lat_total = self.mesh.lats.shape[0] // self.patch_size[1] - 2*self.lat_segment_size
            self.resolution = (8, self.lat_total, self.mesh.lons.shape[0] // self.patch_size[2])

        if self.use_matepoint:
            self.checkpointfn = matepoint.checkpoint
        else: self.checkpointfn = checkpoint.checkpoint
        #self.checkpointfn = None

        ## legacy shit
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
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition_3d(x, window_size):
    B, Z, H, W, C = x.shape

    dimprint("3dinp", x.shape)
    # Reorganize data to calculate window attention
    x_window = x.view(B, Z // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)

    x_window = x_window.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], C)
    dimprint("3dout", x_window.shape)

    return x_window

def window_reverse_3d(windows, window_size, Z, H, W):
    """
    Args:
        windows: (num_windows*B, window_size0, window_size1, window_size2, C)
        window_size (int): Window size
        Z (int): Depth of image
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, Z, H, W, C)
    """
    B = int(windows.shape[0] / (Z * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, Z // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Z, H, W, -1)
    return x

class WindowAttention3D(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, Mlev, Mlat, Mlon, earth_specific=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., FLASH=True):

        super().__init__()
        self.FLASH = FLASH
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.earth_specific = earth_specific

        self.Mlev = Mlev
        self.Mlat = Mlat
        self.Mlon = Mlon

        # define a parameter table of relative position bias
        if self.earth_specific and 1:
            self.earth_bias_table = nn.Parameter(
                torch.zeros(window_size[0]**2 * window_size[1]**2 * (2 * window_size[2] - 1), Mlev, Mlat, num_heads))

            #coords_lev = torch.arange(Mlev)
            #coords_lat = torch.arange(Mlat)

            #coords_lon = torch.zeros(Mlon, dtype=coords_lat.dtype)

            # get pair-wise relative position index for each token inside the window
            coords_zi = torch.arange(self.window_size[0])
            coords_zj = -torch.arange(self.window_size[0]) * self.window_size[0]

            coords_hi = torch.arange(self.window_size[1])
            coords_hj = -torch.arange(self.window_size[1])*self.window_size[1]
       
            coords_w = torch.arange(self.window_size[2])
            coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
            coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))

            coords_1_flat = torch.flatten(coords_1, 1)
            coords_2_flat = torch.flatten(coords_2, 1)
            coords = coords_1_flat[:, :, None] - coords_2_flat[:, None, :]
            #print("aaa", coords.shape)
            coords = coords.permute(1, 2, 0)
            #print("yoo coords", coords.shape)
            #print(coords, len(set(list(coords.flatten().numpy().astype(np.int64)))), coords.max(), coords.min())
            coords[:, :, 2] += self.window_size[2] - 1
            coords[:, :, 1] *= 2 * self.window_size[2] - 1
            coords[:, :, 0] *= (2 * self.window_size[2] - 1)*self.window_size[1]*self.window_size[1]

            relative_position_index = coords.sum(-1)#.to(torch.int16)
            relative_position_index = torch.cat([relative_position_index[:,:,np.newaxis] for _ in range(Mlon)], axis=-1)
            dimprint("RELATIVE POSITION INDEX", relative_position_index.shape, coords.shape)
            dimprint("uniq", len(set(list(relative_position_index.flatten().numpy()))), self.earth_bias_table.shape)
            #exit()

            """

            coords = torch.stack(torch.meshgrid([coords_lat, coords_lon, coords_h, coords_w]))  # 2, Wh, Ww
            #print("uhih", coords.shape)
            coords_flatten = torch.flatten(coords, 3)  # 2, Wh*Ww
            #print("uhih", coords_flatten.shape)
            #print("mlat", Mlat, "mlon", Mlon)
            relative_coords = coords_flatten[:, :, :, :, None] - coords_flatten[:, :, :, None, :]  # 2, Wh*Ww, Wh*Ww
            #print("hmmm", relative_coords.shape)
            relative_coords = relative_coords.permute(1, 2, 3, 4, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            #print("hmmm", relative_coords.shape)
            relative_coords[:, :,  :, :, 2] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, :, :, 3] += self.window_size[1] - 1
            relative_coords[:, :, :, :, 2] *= 2 * self.window_size[1] - 1
            for i in range(Mlat):
                relative_coords[i,:,:,:,0] += i * (2*window_size[0]-1)*(2*window_size[1]-1)
            print(relative_coords.shape)
            print("hullo")
            exit()
            """
            """
            print(relative_coords[1])
            print(relative_coords.shape)
            exit()
            print("heyo uh", relative_coords.shape)
            """
            #relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            #print("heyo uh2", relative_position_index.shape)

        else:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            #print("coordS", coords.shape)
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            #print("coordS_fl", coords_flatten.shape)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        #print("heyuh", relative_coords.shape, "index", relative_position_index.shape)
        #print("relative coords", relative_coords)
        #print("relative pos", relative_position_index)
        self.register_buffer("relative_position_index", relative_position_index)
        dimprint("RELATIVE POSITION INDEX", self.relative_position_index.shape)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.earth_specific:
            trunc_normal_(self.earth_bias_table, std=.02)
        else:
            trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, mask=None,passthru=False):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        if passthru:
            return x

        B_, N, C = x.shape
        assert C == self.dim

        dimprint("earth_specific:", self.earth_specific)
        dimprint(f"Window Attn dim: dim:{self.dim}, head:{self.num_heads}")
        dimprint("Window Attn X:", x.shape)


        if not self.FLASH:
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            dimprint("qkv shape", qkv.shape, "BNC", x.shape)
            sizeprint(qkv,'qkv')
            
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)


            q.mul_(self.scale)

            attn = (q @ k.transpose(-2, -1))

            dimprint("attn shape", attn.shape)
            sizeprint(attn,'attn')


            if self.earth_specific:
                dimprint("attention shape is", attn.shape)
                dimprint("eyoooo")

                sq = self.window_size[0] * self.window_size[1] * self.window_size[2]
                

                #earth_position_bias = self.earth_bias_table[:, self.relative_position_index.view(-1)].view(self.N, sq, sq, self.num_heads)
                #print("hi", self.earth_bias_table.shape, self.relative_position_index.shape, self.relative_position_index.view(-1).shape)
                #aa = self.relative_position_index.view(-1).cpu().numpy()
                #print("uniq", len(set(list(aa))))

                dimprint("earth_bias_table", self.earth_bias_table.shape)

                #earth_position_bias = self.earth_bias_table[self.relative_position_index.view(-1)].view(self.Mlat*self.Mlon, sq, sq, self.num_heads)
                earth_position_bias = self.earth_bias_table[self.relative_position_index.view(-1)].view(sq, sq, self.Mlon, self.Mlev, self.Mlat, self.num_heads)
                sizeprint(earth_position_bias,'earth_position_bias')
                

                # reshaping and permuting epb makes reserved memory to jump by an exta gig. Not sure why, could be opimtized if a problem in future
                earth_position_bias = earth_position_bias.permute(3, 4, 2, 5, 0, 1).flatten(start_dim=0, end_dim=2)
                dimprint("Epb now is", earth_position_bias.shape)
                sizeprint(earth_position_bias,'earth_position_bias')

                BB = attn.shape[0] // earth_position_bias.shape[0]
                assert attn.shape[0] % earth_position_bias.shape[0] == 0
                earth_position_bias = torch.cat([earth_position_bias for _ in range(BB)])
                #earth_position_bias = self.dummy1(earth_position_bias)
                attn = attn + earth_position_bias
                #attn = self.dummy2(attn)
                #exit()

            else:
                #print("hi", self.relative_position_bias_table.shape, self.relative_position_index.shape, self.relative_position_index.view(-1).shape)
                relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
                #print("hello", relative_position_bias.shape)
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0)


            """
            print("rel position bias", relative_position_bias.shape)
            print("rel position bias2", relative_position_bias.unsqueeze(0).shape)
            print("earth", self.earth_bias_table.shape)
            print("earth2", self.earth_bias_table[:, self.relative_position_index.view(-1)].shape)
            """
            #print("test", earth_position_bias.shape, attn.shape, earth_position_bias.unsqueeze(0).shape)
            #print("attn", attn.shape)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                #mask = self.dummy3(mask)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)

            attn = self.attn_drop(attn)

            attn_v = attn @ v

        else:
            # TODO: this had a lot of asserts removed

            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            sq = self.window_size[0] * self.window_size[1] * self.window_size[2]

            earth_position_bias = self.earth_bias_table[self.relative_position_index.view(-1)].view(sq, sq, self.Mlon, self.Mlev, self.Mlat, self.num_heads)

            # reshaping and permuting epb makes reserved memory to jump by an exta gig. Not sure why, could be opimtized if a problem in future
            earth_position_bias = earth_position_bias.permute(3, 4, 2, 5, 0, 1).flatten(start_dim=0, end_dim=2)

            #BB = 1
            BB = x.shape[0] // earth_position_bias.shape[0]
            assert x.shape[0] % earth_position_bias.shape[0] == 0
            #print("yooo what", earth_position_bias.shape, q.shape, k.shape, v.shape, "x", x.shape, "B_", B_, self.Mlat, self.Mlev, self.Mlon)
            #assert attn.shape[0] % earth_position_bias.shape[0] == 0
            earth_position_bias = torch.cat([earth_position_bias for _ in range(BB)])
            earth_position_bias = earth_position_bias.contiguous()

            if mask is not None:
                #print("masky mask", mask.shape)
                mask = torch.cat([mask for _ in range(BB)])
                earth_position_bias += mask.unsqueeze(1)
            else:
                pass

            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                attn_v = torch.nn.functional.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), earth_position_bias)



        x = attn_v.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Swin3DTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=None, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False,attn_mask = None, FLASH=True, perturber=0):
        super().__init__()
        assert window_size is not None
        self.dim = dim
        self.perturber = perturber
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        dimprint("[SHIFT]",shift_size)
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        for a,b in zip(self.input_resolution, self.window_size):
            assert a >= b
            assert a % b == 0
        #assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads, Mlev=input_resolution[0]//window_size[0], Mlat=input_resolution[1]//window_size[1], Mlon=input_resolution[2]//window_size[2],
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, FLASH=FLASH)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn_mask = attn_mask
        #print("attn mask", attn_mask.data_ptr()) 
        #self.register_buffer("attn_mask", attn_mask)
        #print("attn mask self.", self.attn_mask.data_ptr()) 
        self.fused_window_process = fused_window_process

    def forward(self, x,**kwargs): 
        Z, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == Z * H * W, "input feature has wrong size"

        dimprint("BLC", B, L, C)
        dimprint("ZHW", Z, H, W)
        dimprint("x", x.shape)

        shortcut = x
        x = self.norm1(x)
        #print("eyooo0", x.shape)
        x = x.view(B, Z, H, W, C)

        #print("eyooo", x.shape)

        # cyclic shift
        if self.shift_size[0] > 0:
            assert not self.fused_window_process, "whoops John nuked the code for this"
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            # partition windows
            x_windows = window_partition_3d(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition_3d(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        dimprint("window_size", self.window_size)

        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size, C

        dimprint("doing attention with", x_windows.shape, Z, H, W, C)
        # W-MSA/SW-MSA


        mask = self.attn_mask if self.shift_size[0] != 0 else None
        attn_windows = self.attn(x_windows, mask=mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)



        # reverse cyclic shift
        if self.shift_size[0] > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse_3d(attn_windows, self.window_size, Z, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2, 3))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size) 
                # [sic]
        else:
            shifted_x = window_reverse_3d(attn_windows, self.window_size, Z, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, Z * H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        if self.perturber != 0:
            x = x + self.perturber * self.drop_path(self.mlp(self.norm2(x)))
        else:
            #assert False
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class BasicLayer3D(nn.Module):
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

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, checkpointfn=matepoint.checkpoint,
                 fused_window_process=False, FLASH=True, checkpoint_every=1, skip_every=None, perturber=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.checkpointfn = checkpointfn
        self.checkpoint_every = checkpoint_every
        self.window_size = window_size
        self.shift_size = [x // 2 for x in window_size]
        assert dim % num_heads == 0, "this assert was added by joan, are you proud of him?"
        # assert dim // num_heads > 16, "Highly sus how small the head dimention is. This assert was added by John, Joan you still have a lot to learn"

        self.skip_every = skip_every

        # build blocks

        # calculate attention mask for SW-MSA
        Z, H, W = self.input_resolution
        img_mask = torch.zeros((1, Z, H, W, 1))  # 1 H W 1
        z_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))

        h_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        w_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cnt = 0
        for z in z_slices:
            for h in h_slices:
                #for w in w_slices:
                img_mask[:, z, h, :, :] = cnt
                cnt += 1

        #print("masked cnt", cnt)
        assert Z % self.window_size[0] == 0, f"Z {Z} window_size {self.window_size[0]}"
        assert H % self.window_size[1] == 0, f"H {H} window_size {self.window_size[1]}"
        assert W % self.window_size[2] == 0, f"W {W} window_size {self.window_size[2]}"

        mask_windows = window_partition_3d(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, np.prod(self.window_size))
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)).to(torch.float16)
        self.register_buffer("attn_mask", attn_mask)
        #test = attn_mask.view(360 // 8, 720 // 8, 64, 64)
        #print(test[-2, :, :, :].sum())
        # TODO: verify that the masking is okay

        
        mlist = []
        #attn_mask = None
        for i in range(depth):
            tb_block = Swin3DTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=tuple([0 if (i % 2 == 0) else x // 2 for x in window_size]),
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process, FLASH=FLASH,
                                 attn_mask = self.attn_mask, perturber=perturber)
            mlist.append(tb_block)
        self.blocks = nn.ModuleList(mlist)
        self.matepoint_ctx = []

    def forward(self, x):
        global Gmatepoint_stream
        if self.skip_every is None:
            #print("attn mask", self.attn_mask.shape, "x", x.shape)
            for i, blk in enumerate(self.blocks):
                print_mem(f"before_block.{i}")
                if self.checkpointfn is not None and i % self.checkpoint_every == 0:
                    x = self.checkpointfn(blk, x, matepoint_ctx=self.matepoint_ctx,stream=Gmatepoint_stream, use_reentrant=False, preserve_rng_state=False)
                else:
                    #print("no checkpoint!!")
                    x = blk(x)#,self.attn_mask)
                print_mem(f"after_block.{i}")  
        else:
            assert False, "john thinks that this codepathway probs shouldnt' be used anymore"
            short = 0
            for i, blk in enumerate(self.blocks):
                if i % self.skip_every == 0:
                    short = x
                ofs = 0
                if (i+1) % self.skip_every == 0:
                    ofs = short
                if self.checkpointfn is not None and i % self.checkpoint_every == 0:
                    x = ofs + self.checkpointfn(blk, x, self.attn_mask if blk.shift_size[0] != 0 else None, stream=Gmatepoint_stream, use_reentrant=False)
                else:
                    x = ofs + blk(x,self.attn_mask if blk.shift_size[0] != 0 else None)
        matepoint_pipeline(self.matepoint_ctx,Gmatepoint_stream)
        return x

    def update_masks(self):
        for i, blk in enumerate(self.blocks):
            blk.attn_mask = self.attn_mask

class ForecastStepBase(EarthSpecificModel,SourceCodeLogger):

    def __init__(self,config):
        self.mesh = config.mesh
        super().__init__(self.mesh)
        self.config = config
        self.radiation_cache = {}
        self.n_addl_vars = 2
        if self.config.neorad:
            self.n_addl_vars += 2 # solar angle
        self.input_dim = self.mesh.n_pr_vars * self.mesh.n_levels + self.mesh.n_sfc_vars
        self.O = self.input_dim
        self.D = self.input_dim + self.n_addl_vars + self.n_const_vars
        self.total_sfc = self.mesh.n_sfc_vars + self.n_addl_vars + self.n_const_vars
        self.total_pr = self.mesh.n_pr_vars * self.mesh.n_levels

        self.neocache = {}

    def interpget(self, src, toy, hr, a=300, b=400):
        if src not in self.neocache:
            self.neocache[src] = {}

        def load(xx, hr):
            if (xx,hr) in self.neocache[src]:
                return self.neocache[src][(xx, hr)]
            #print("loading", src, xx, hr)
            ohp = torch.HalfTensor(((np.load(CONSTS_PATH+'/%s/%d_%d.npy' % (src, xx, hr)) - a) / b))
            self.neocache[src][(xx,hr)] = ohp
            return ohp
        if self.config.neorad_subsamp and toy % 2 == 1:
            avg = load((toy-1)%366, hr) * 0.5 + load((toy+1)%366, hr) * 0.5
        else:
            avg = load(toy, hr)
        return avg

    
    def get_radiation(self, toy, hr=0):
        if self.config.neorad:
            return self.interpget("neoradiation_%d"%self.mesh.subsamp, toy, hr)

        if toy in self.radiation_cache:
            return self.radiation_cache[toy]
        else:
            rad = (torch.HalfTensor(np.load(CONSTS_PATH+'/radiation_%d/%d.npy' % (self.mesh.subsamp, toy))) - 300) / 400
            self.radiation_cache[toy] = rad
            return rad  

    def add_additional_vars(self,x,t0s):
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
                print(f"Failed to load solarangle_{self.mesh.subsamp} for {toy} {date} {date.hour}", flush=True) # [sic]
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
        assert xsfc.shape[-1] == self.total_sfc
        xsfc = xsfc.permute(0, 3, 1, 2)
        return xpr, xsfc
    
    @staticmethod
    def combine_pr_sfc(xpr_conv, xsfc_conv, c):
        xpr_conv = xpr_conv.permute(0, 2, 3, 4, 1)
        xsfc_conv = xsfc_conv.permute(0, 2, 3, 1)
        if c.surface_sandwich_bro:
            x_conv = torch.cat((xsfc_conv[:, np.newaxis], xpr_conv, xsfc_conv[:, np.newaxis]), axis=1)
        else:
            x_conv = torch.cat((xpr_conv, xsfc_conv[:, np.newaxis]), axis=1)
        x_conv = torch.flatten(x_conv, start_dim=1, end_dim=3)
        return x_conv
    
    def to(self, *args, **kwargs):
        global Gmatepoint_stream
        super().to(*args, **kwargs)
        #device = kwargs.get('device', args[0] if args else 'cuda' if torch.cuda.is_available() else 'cpu')
        Gmatepoint_stream = torch.cuda.Stream()
        for swin in ['enc_swin','dec_swin','proc_swin']:
            if type(getattr(self,swin,None)) == BasicLayer3D: 
                getattr(self,swin).update_masks()
        if hasattr(self,'decoders'):
            try:
                for k,v in self.decoders.items():
                    v.to(*args, **kwargs)
            except:
                self.singledec.to(*args, **kwargs)
        return self
    
    def rollout(self, xs, time_horizon=None,dt_dict=None,min_dt=3, callback=None, ic_callback=None, dts=None):
        """
        Makes predictions going out to the time horizon in hours
        Does so at the max resolution the model can support
        For example, with a target time horizon of 48h, and support for 12h and 24h timestamps, it would generate 12, 24, 24 + 12, 24 + 24

        callback can be passed to get intermediate results, which can save memory

        dt_dict is a schedule of time resolutions, defined as start_forecast_hour: resolution
        For example, {6:3,24:6,72:12} would give min_dt until hour 6, 3 hour resolution until hour 24, etc
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

        progress = [SimpleNamespace(target_dt=dtt,tensors=xs,accum_dt=0,steps_left=min_additions(ts,dtt)) for dtt in dts]
        finished = {}
        while progress:
            min_t = np.min([v.accum_dt for v in progress])
            sub = [v for v in progress if v.accum_dt == min_t]
            step = sub[0].steps_left[0]
            sub = [v for v in sub if v.steps_left[0] == step]
            print(f"STEP step_from: {sub[0].accum_dt} step: {step} progress={len(progress)}")
            ic_callback = ic_callback if len(progress) == len(dts) else None 
            if ic_callback is None:
                new_tensors = [self.forward(all_to(sub[0].tensors,'cuda'),dt=step,gimme_deltas=False), sub[0].tensors[-1] + step * 3600]
            else:
                new_tensors = [self.forward(all_to(sub[0].tensors,'cuda'),dt=step,gimme_deltas=False,ic_callback=ic_callback), sub[0].tensors[-1] + step * 3600]
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

class ForecastStepEncoder(ForecastStepBase):
    def __init__(self,config=None):
        super().__init__(config)
        self.dummmy = nn.Linear(5,5)
        c = self.config

        H = c.preconv_hidden_dim
        C = c.patch_size[0] 
        Z = c.resolution[0] 
        idx = np.searchsorted(levels_medium,c.mesh.levels)
        bin = [x // C for x in idx]
        print(bin)
        level_encoders = []
        level_encoder_idxs = []
        for i in range(Z-1):
            idxs = [j for j, x in enumerate(bin) if x == i]
            m = nn.Linear(len(idxs)*c.mesh.n_pr_vars,H)
            level_encoders.append(m)
            level_encoder_idxs.append(idxs)
        self.level_encoders = nn.ModuleList(level_encoders)
        self.level_encoder_idxs = level_encoder_idxs 
        self.surface2all_encoder = nn.Linear(self.total_sfc,H)
        self.surface_encoder = nn.Linear(self.total_sfc,H)

        patch = (1, c.patch_size[1], c.patch_size[2])
        self.conv = nn.Conv3d(in_channels=c.preconv_hidden_dim, out_channels=c.hidden_dim, kernel_size=patch, stride=patch)

    def forward(self, xs, dt=None):
        c = self.config

        x,t0 = xonly(xs)

        B, Nlat, Nlon, D = x.shape

        xp, xs = self.breakup_pr_sfc(x, t0)
        H = c.preconv_hidden_dim
        Z = c.resolution[0]

        xenc = torch.zeros((B, Z, Nlat, Nlon, H), device=xp.device, dtype=xp.dtype)
        xs_enc = self.surface2all_encoder(xs.permute(0,2,3,1))
        for i,encoder in enumerate(self.level_encoders):
            idxs = self.level_encoder_idxs[i]
            xenc[:,i] = encoder(xp[:,:,idxs].permute(0,3,4,1,2).flatten(start_dim=3))
            xenc[:,i] += xs_enc
        xenc[:,-1] = self.surface_encoder(xs.permute(0,2,3,1))
        xconv = self.conv(xenc.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        xconv = torch.flatten(xconv, start_dim=1, end_dim=3)
        return xconv

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
            self.dec_swin = BasicLayer3D(dim=H, input_resolution=c.resolution, depth=c.dec_swin_depth, num_heads=num_heads, window_size=c.window_size, FLASH=c.FLASH, drop_path=c.drop_path, checkpoint_every=c.checkpoint_every, checkpointfn=c.checkpointfn)#, perturber=c.perturber)
        else: 
            self.dec_swin = nn.Identity()

        if c.lat_compress:
            self.deconv_lat = nn.ConvTranspose3d(out_channels=c.conv_dim, in_channels=H, kernel_size=(1, 2, 1), stride=(1, 2, 1))
            self.deconv_sfc_lat = nn.ConvTranspose2d(out_channels=c.conv_dim, in_channels=H, kernel_size=(2, 1), stride=(2, 1))


        if c.surface_sandwich_bro:
            self.deconv = nn.ConvTranspose3d(out_channels=c.output_mesh.n_pr_vars, in_channels=H, kernel_size=c.patch_size, stride=c.patch_size,output_padding=(1,0,0)) 
        else:
            self.deconv = nn.ConvTranspose3d(out_channels=c.output_mesh.n_pr_vars, in_channels=H, kernel_size=c.patch_size, stride=c.patch_size)
        #pnew = (4, c.patch_size[1], c.patch_size[2]) # XXX HERE!
        #print(pnew,c.patch_size)

        self.deconv_sfc = nn.ConvTranspose2d(out_channels=c.output_mesh.n_sfc_vars, in_channels=H, kernel_size=c.patch_size[1:], stride=c.patch_size[1:])

        if c.output_deltas:
            delta_norm_matrix = torch.from_numpy(load_delta_norm(dt, c.output_mesh.n_levels, c.output_mesh)[1])
            self.register_buffer("delta_norm_matrix", delta_norm_matrix)

        self.H = H            
    
    def forward(self, x):
        B = x.shape[0]
        c = self.config
        x_tr = self.dec_swin(x)
        dimprint("x_tr", x_tr.shape)

        x_tr = x_tr.permute(0, 2, 1)
        x_tr = x_tr.view(B, self.H, c.resolution[0], c.resolution[1], c.resolution[2])
        dimprint("X_tr", x_tr.shape)

        if c.surface_sandwich_bro:
            y_pr_conv = x_tr[:, :, 1:-1, :, :]
        else:
            y_pr_conv = x_tr[:, :, :-1, :, :]
        y_sfc_conv = x_tr[:, :, -1, :, :]
        dimprint("ysconv", y_pr_conv.shape, y_sfc_conv.shape)

        if c.lat_compress:
            nshape = list(y_pr_conv.shape)
            n1 = c.lat_segment_size
            nshape[-2] = c.post_conv_nlat
            nx = torch.zeros(nshape, device=y_pr_conv.device, dtype=y_pr_conv.dtype)
            nx[...,:n1*2,:] = self.deconv_lat(y_pr_conv[...,:n1,:])
            nx[...,n1*2:-n1*2,:] = y_pr_conv[...,n1:-n1,:]
            nx[...,-n1*2:,:] = self.deconv_lat(y_pr_conv[...,-n1:,:])
            del y_pr_conv
            y_pr_conv = nx
            nshape = list(y_sfc_conv.shape)
            nshape[-2] = c.post_conv_nlat
            nx = torch.zeros(nshape, device=y_sfc_conv.device, dtype=y_sfc_conv.dtype)
            nx[...,:n1*2,:] = self.deconv_sfc_lat(y_sfc_conv[...,:n1,:])
            nx[...,n1*2:-n1*2,:] = y_sfc_conv[...,n1:-n1,:]
            nx[...,-n1*2:,:] = self.deconv_sfc_lat(y_sfc_conv[...,-n1:,:])
            del y_sfc_conv
            y_sfc_conv = nx

        y_pr = self.deconv(y_pr_conv)
        y_sfc = self.deconv_sfc(y_sfc_conv)

        y_sfc = y_sfc.permute(0, 2, 3, 1)
        dimprint("ys", y_pr.shape, y_sfc.shape)
        y_pr = y_pr.permute(0, 3, 4, 1, 2)
        y_pr = torch.flatten(y_pr, start_dim=-2)
        dimprint("ypr", y_pr.shape)
        y = torch.cat((y_pr, y_sfc), axis=-1)
        dimprint("eyoo y", y.shape)
        dimprint("\n\n\nXXXXXXXXX\n\n\n")
        return y

class ForecastStepSwin3D(ForecastStepBase):

    def __init__(self, config):
        self.config = config
        c = self.config
        super().__init__(config)

        self.do_sub = False

        self.last_steps = None
        #assert not self.input_deltas


        #self.encoder = nn.Sequential(
        #        nn.Linear(c.D, c.hidden_dim),
        #)

        if c.lat_compress:
            self.conv_lat = nn.Conv3d(in_channels=c.conv_dim, out_channels=c.conv_dim, kernel_size=(1, 2, 1), stride=(1, 2, 1))
            self.conv_sfc_lat = nn.Conv2d(in_channels=c.conv_dim, out_channels=c.conv_dim, kernel_size=(2, 1), stride=(2, 1))

        #if drop_path is None or drop_path == True:
        #    drop_path = list(np.linspace(0, 0.2, depth))

        self.AAAA = 0
        if not self.AAAA:
            self.conv = nn.Conv3d(in_channels=self.mesh.n_pr_vars, out_channels=c.conv_dim, kernel_size=c.patch_size, stride=c.patch_size)
            self.conv_sfc = nn.Conv2d(in_channels=self.total_sfc, out_channels=c.conv_dim, kernel_size=c.patch_size[1:], stride=c.patch_size[1:])
        else:
            self.encoder = ForecastStepEncoder(config=self.config)

        if c.enc_swin_depth > 0:
            self.enc_swin = BasicLayer3D(dim=c.hidden_dim, input_resolution=c.resolution, depth=c.enc_swin_depth, num_heads=c.num_heads, window_size=c.window_size, FLASH=c.FLASH, drop_path=c.drop_path, checkpointfn=c.checkpointfn)#, perturber=c.perturber)
        else:
            self.enc_swin = nn.Identity()

        self.proc_swin = BasicLayer3D(dim=c.hidden_dim, input_resolution=c.resolution, depth=c.proc_swin_depth, num_heads=c.num_heads, window_size=c.window_size, FLASH=c.FLASH, drop_path=c.drop_path, checkpointfn=c.checkpointfn, perturber=c.perturber)

        if not c.output_deltas:
            self.singledec = ForecastStepDecoder(dt=0, config=self.config)
            class ihmlawtd:
                def __init__(lol): pass
                def __getitem__(lol, key): return self.singledec
            self.decoders = ihmlawtd()
            #self.decoders = {}
            #for t in c.timesteps:
            #    self.decoders[str(t)] = self.singledec
        else:
            self.decoders = nn.ModuleDict()
            for t in c.timesteps:
                self.decoders[str(t)] = ForecastStepDecoder(dt=t, config=self.config)

        if c.decoder_reinput_initial:
            self.downsample_latent = nn.Linear(c.conv_dim, c.decoder_reinput_size)




    def conv_forward(self, x, t0s):
        c = self.config
        xpr,xsfc = self.breakup_pr_sfc(x, t0s)
        #print("previously", torch.mean(xsfc[0, c.mesh.n_sfc - len(c.mesh.extra_sfc_vars):c.mesh.n_sfc].float(), axis=(1,2)))
        xsfc[:, c.mesh.n_sfc - len(c.mesh.extra_sfc_vars):c.mesh.n_sfc] = 0
        #assert len(c.mesh.extra_sfc_vars) == 0
        #print("after", torch.mean(xsfc[0, c.mesh.n_sfc - len(c.mesh.extra_sfc_vars):c.mesh.n_sfc].float(), axis=(1,2)))
        #print("after", np.mean(xsfc[c.mesh.n_sfc - len(c.mesh.extra_sfc_vars):c.mesh.n_sfc].cpu().numpy().astype(np.float32), axis=(0,2,3)))
        xpr_conv = self.conv(xpr)
        xsfc_conv = self.conv_sfc(xsfc)
        
        if c.lat_compress:
            nshape = list(xpr_conv.shape)
            lat_full_size = xpr_conv.shape[-2]
            n1 = c.lat_segment_size
            nshape[-2] = c.lat_total
            nx = torch.zeros(nshape, device=xpr_conv.device, dtype=xpr_conv.dtype)
            nx[...,:n1,:] = self.conv_lat(xpr_conv[...,:n1*2,:]) 
            nx[...,n1:-n1,:] = xpr_conv[...,n1*2:-n1*2,:]
            nx[...,-n1:,:] = self.conv_lat(xpr_conv[...,-n1*2:,:])
            del xpr_conv
            xpr_conv = nx
            nshape = list(xsfc_conv.shape)
            nshape[-2] = c.lat_total
            nx = torch.zeros(nshape, device=xsfc_conv.device, dtype=xsfc_conv.dtype)
            nx[...,:n1,:] = self.conv_sfc_lat(xsfc_conv[...,:n1*2,:])
            nx[...,n1:-n1,:] = xsfc_conv[...,n1*2:-n1*2,:]
            nx[...,-n1:,:] = self.conv_sfc_lat(xsfc_conv[...,-n1*2:,:])
            del xsfc_conv
            xsfc_conv = nx
 
        x_conv = ForecastStepBase.combine_pr_sfc(xpr_conv, xsfc_conv, self.config)

        return x_conv

    def forward_inner(self, xs, dt=None,skip_enc=False):
        if hasattr(self, 'output_deltas') and hasattr(self.config, 'output_deltas'):
            assert self.output_deltas == self.config.output_deltas, "Ohp, Joan I changed output deltas to be in the config"

        #x,t0 = xonly(xs)
        x, t0s = xs
        c = self.config
        if dt is None: dt = 24
        assert dt in c.timesteps, f"dt {dt} not in {c.timesteps}"

        if c.return_random:
            assert 0
            #return torch.randn((B,Nlat,Nlon,c.O), device=x.device, dtype=x.dtype)

        if not skip_enc:
            if self.AAAA:
                assert len(c.mesh.output_only_vars) == 0, "you gotta zero out shit here bro"
                x_conv = self.encoder(xs)
                #x_conv = c.checkpointfn(self.encoder, x, use_reentrant=False, preserve_rng_state=False)
            else: 
                x_conv = self.conv_forward(x,t0s)
        else:
            x_conv = x
            assert len(c.mesh.output_only_vars) == 0, "you gotta zero out shit here bro"

        x_tr = self.enc_swin(x_conv)
        x_initial = x_tr
        
        assert dt % c.processor_dt == 0
        for i in range(dt // c.processor_dt):
            if c.delta_processor:
                x_tr = x_tr + self.proc_swin(x_tr)
            else:
                x_tr = self.proc_swin(x_tr)
            #x_tr = checkpoint.checkpoint(self.proc_swin, x_tr, use_reentrant=False)

        if c.decoder_reinput_initial:
            extra = self.downsample_latent(x_initial)
            x_tr = torch.cat((x_tr, extra), dim=-1)

        decoder = self.decoders[str(dt)]
        y = decoder(x_tr)
        return y
 

    def forward(self, xs, dt=None,skip_enc=False,gimme_deltas=False,save_intermediates=False):
        _,t0s = xs
        if dt in self.config.timesteps:
            y = self.forward_inner(xs,dt=dt,skip_enc=skip_enc)
            if self.config.output_deltas and not gimme_deltas:
                y = xs[0] + y * self.decoders[str(dt)].delta_norm_matrix
                y = set_metadata(y,t0s+3600*dt) 
            else: 
                y = set_metadata(y,t0s+3600*dt) 
                y.meta.delta_info = dt
            self.last_steps = [int(dt)]
            return y
        #assert False, "fuck vectorized metadata part 2"
        steps = min_additions(self.config.timesteps, dt)
        #print(f'model is doing multistep, {steps}')
        assert steps is not None, "No combination of timesteps will work for this"
        assert len(self.config.inputs) == 1, "multistep only works with one input for now"
        assert not gimme_deltas, "gimme_deltas not supported for multistep"
        if save_intermediates: self.last_intermediates = []
        total_dt = 0
        for i,step in enumerate(steps):
            #print("steppety step", i, step, steps)
            y = self.forward_inner(xs,dt=step,skip_enc=skip_enc)
            #y = checkpoint.checkpoint(self.forward_inner, xs)
            if self.config.output_deltas:
                y = xs[0] + y * self.decoders[str(step)].delta_norm_matrix
            #print("uh oh what the fuck", dt, total_dt, "step", step, steps)
            total_dt += step
            y = set_metadata(y,t0s+3600*total_dt)
            xs = [y,xs[1] + step * 3600]
            if save_intermediates: self.last_intermediates.append((y,total_dt))
        assert total_dt == dt, "wtf is going on in here"
        assert t0s + dt * 3600 == xs[-1]
        self.last_steps = steps
        return y
    
    def change_droppath_prob(self, prob):
        for name, module in self.named_modules():
            if isinstance(module, DropPath):
                module.drop_prob = prob

class ForecastStepCombo(ForecastStepBase):
    def __init__(self, models, adapter=None):
        self.config = copy.deepcopy(models[0].config)
        super().__init__(self.config)
        all_timesteps = []
        for m in models:
            all_timesteps += m.config.timesteps
        print(all_timesteps)
        all_timesteps = sorted(list(np.unique(all_timesteps)))
        for m1 in models:
            outputs = m1.config.outputs + self.config.outputs
            inputs = m1.config.inputs + self.config.inputs
            for i in inputs:
                for o in outputs:
                    x = to_mesh(0,o,i,check_only=True)
        self.config.timesteps = all_timesteps
        if adapter is not None:
            self.config.inputs = adapter.config.inputs
            to_mesh(0,adapter.config.outputs[0],models[0].config.inputs[0],check_only=True)
        self.adapter = adapter
        self.models = nn.ModuleList(models)

    def forward(self, xs, dt=None, gimme_deltas=False, ic_callback=None):
        assert dt is not None, "dt must be specified for combo model"
        #We find which model has this timestep, and also has the largest timestep internally
        model_i = np.argmax([max(m.config.timesteps) if dt in m.config.timesteps else 0 for m in self.models])
        model = self.models[model_i]
        xs = self.adapt(xs, model_i)

        print(f"Using {model.name} for timestep {dt}")
        y = model.forward(xs, dt=dt, gimme_deltas=False)
        #, gimme_deltas=True)
        #y = origx + y * model.decoders[str(dt)].delta_norm_matrix
        #y = set_metadata(y,xs[-1]+3600*dt) 
        print(y.shape)
        if model_i != 0:
            y = to_mesh(y, model.config.outputs[0], self.models[0].config.outputs[0], fill_with_zero=True)
        if ic_callback:
            ic_callback(xs,y)

        return y
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for m in self.models:
            m.to(*args, **kwargs)
        return self

    def adapt(self, xs, model_i=None):
        assert model_i is not None, "Set the index of model to be used to check mesh compatibility"
        model = self.models[model_i]

        if self.adapter is not None:
            assert len(self.adapter.config.inputs) != len(model.config.inputs), "if adapter and model take same number of inputs, this code can't tell what to used, needs updating"
            if len(xs) == len(self.adapter.config.inputs)+1:
                #this means it must need the adapter
                #print(f"Using adapter {self.adapter.name}")
                #print("hey", [torch.mean(y, axis=(0,1,2))[0] for y in xs[:-1]])
                #print("sources", [src.source for src in self.adapter.config.inputs])
                #xs = [xs[1], xs[0], xs[2]]
                xadapted = self.adapter(xs,gimme_deltas=False)
                #rms = torch.sqrt(torch.mean(torch.square((xadapted - xs[0]))))
                #print("adapter rms", rms)
                #print("skipping adapter actually!", len(xs), xs[0].shape, xs[1].shape)
                #xadapted = xs[1]
                #print(xadapted.shape, xs[0].shape, xs[1].shape)
                #xadapted = torch.clamp(xadapted, -5, 5)
                xs = [xadapted] + [xs[-1]]
                if self.adapter.config.outputs[0].full_varlist != model.config.inputs[0].full_varlist:
                    xs[0] = to_mesh(xs[0], self.adapter.config.outputs[0], model.config.inputs[0])
            else:
                pass#origx = xs[0]
        else:
            assert len(xs) == len(model.config.inputs)+1, f'len(xs) {len(xs)} len(model.config.inputs) {len(model.config.inputs)}'
            #origx = xs[0]
        if model_i != 0:
            xs[0] = to_mesh(xs[0], self.models[0].config.inputs[0], model.config.inputs[0])

        return xs