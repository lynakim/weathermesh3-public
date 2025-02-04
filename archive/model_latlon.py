
from utils import *
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
#from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.layers import DropPath, to_2tuple, trunc_normal_

torch.backends.cuda.enable_flash_sdp(True)


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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
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

    def __init__(self, dim, window_size, num_heads, Mlat, Mlon, earth_specific=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.earth_specific = earth_specific

        self.Mlat = Mlat
        self.Mlon = Mlon

        # define a parameter table of relative position bias
        if self.earth_specific and 1:
            self.earth_bias_table = nn.Parameter(
                torch.zeros(Mlat * (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            coords_lat = torch.arange(Mlat)
            coords_lon = torch.zeros(Mlon, dtype=coords_lat.dtype)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
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
            """
            print(relative_coords[1])
            print(relative_coords.shape)
            exit()
            print("heyo uh", relative_coords.shape)
            """
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
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
        assert relative_position_index.min() >= 0
        assert relative_position_index.max() < 2**15
        relative_position_index = relative_position_index.to(torch.int16)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.earth_specific:
            trunc_normal_(self.earth_bias_table, std=.02)
        else:
            trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.dummy1 = nn.Identity()
        self.dummy2 = nn.Identity()
        self.dummy3 = nn.Identity()
        self.dummy4 = nn.Identity()

    def forward(self, x, mask=None,passthru=False):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        if passthru:
            return x

        B_, N, C = x.shape

        NEO = True

        if NEO:
            #torch.cuda.synchronize()
            #t0 = time.time()

            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            sq = self.window_size[0] * self.window_size[1]
            aa = self.relative_position_index.view(-1).cpu().numpy()
            earth_position_bias = self.earth_bias_table[self.relative_position_index.to(torch.int).view(-1)].view(self.Mlat*self.Mlon, sq, sq, self.num_heads)
            earth_position_bias = earth_position_bias.permute(0, 3, 1, 2)

            BB = 1
            earth_position_bias = torch.cat([earth_position_bias for _ in range(BB)])

            if mask is not None:
                earth_position_bias += mask.unsqueeze(1)
                """
                nW = mask.shape[0]
                print("uhH", mask.shape, earth_position_bias.shape)
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                mask = self.dummy3(mask)
                attn = self.softmax(attn)
                """
            else:
                pass#attn = self.softmax(attn)

            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                attn_v = torch.nn.functional.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), earth_position_bias.contiguous())
            #del q, k, v, qkv, earth_position_bias
            #torch.cuda.empty_cache()
        else:

            #torch.cuda.synchronize()
            #dt = time.time()-t0
            #t0 = time.time()
            #print("neo", dt)

            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            sq = self.window_size[0] * self.window_size[1]
            aa = self.relative_position_index.view(-1).cpu().numpy()
            earth_position_bias = self.earth_bias_table[self.relative_position_index.view(-1)].view(self.Mlat*self.Mlon, sq, sq, self.num_heads)
            earth_position_bias = earth_position_bias.permute(0, 3, 1, 2)

            BB = attn.shape[0] // earth_position_bias.shape[0]
            assert attn.shape[0] % earth_position_bias.shape[0] == 0
            earth_position_bias = torch.cat([earth_position_bias for _ in range(BB)])
            earth_position_bias = self.dummy1(earth_position_bias)
            attn = attn + earth_position_bias
            attn = self.dummy2(attn)
            
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                mask = self.dummy3(mask)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)

            #Here!!!
            attn = self.attn_drop(attn)

            attn_v = attn @ v

            #torch.cuda.synchronize()
            #dt = time.time()-t0
            #print("old", dt, torch.sqrt(torch.mean(torch.square((attn_v - neo_attn_v)))), torch.sqrt(torch.mean(torch.square((attn_v)))))

        x = attn_v.transpose(1, 2).reshape(B_, N, C)


        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
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

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, Mlat=input_resolution[0]//window_size, Mlon=input_resolution[1]//window_size,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            #print("masked cnt", cnt)
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            #print("heyooo", attn_mask.shape)
            #test = attn_mask.view(360 // 8, 720 // 8, 64, 64)
            #print(test[-2, :, :, :].sum())
            # TODO: verify that the masking is okay
            #exit()
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            assert not self.fused_window_process, "whoops John nuked the code for this"
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        
        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x


        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #x = x + self.drop_path(checkpoint.checkpoint(self.mlp, self.norm2(x)))


        return x



class BasicLayer(nn.Module):
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
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=1,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        assert dim % num_heads == 0, "this assert was added by joan, are you proud of him?"

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            #blk.reporter = self.reporter
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

class ForecastStepSwin(nn.Module, SourceCodeLogger):

    def __init__(self, grid=None, transformer_input=384, encoder_hidden=1024, num_heads=16):
        super().__init__()
        self.grid = grid
        self.input_dim = grid.n_pr_vars * grid.n_levels + grid.n_sfc_vars
        self.O = self.input_dim
        self.D = self.input_dim + self.grid.xpos.shape[-1] + 2

        #self.separate_inputs = False
        self.input_deltas = True
        self.output_deltas = True

        """
        self.encoder = nn.Sequential(
                nn.Linear(self.D, transformer_input),
        )


        self.decoder = nn.Sequential(
                nn.Linear(transformer_input, self.input_dim)
        )
        """
        self.encoder = nn.Sequential(
                nn.Linear(self.D, encoder_hidden),
                nn.SiLU(),
                nn.Linear(encoder_hidden, transformer_input),
        )


        self.decoder = nn.Sequential(
                nn.Linear(transformer_input, encoder_hidden),
                nn.SiLU(),
                nn.Linear(encoder_hidden, self.input_dim)
        )

        self.swin1 = BasicLayer(dim=transformer_input, input_resolution=(360, 720), depth=7, num_heads=num_heads, window_size=8)


        return

    def forward(self, x):
        #self.swin1.reporter = self.reporter
        B,Nlat,Nlon,D = x.shape

        x = torch.flatten(x, start_dim=1, end_dim=2)


        x_enc = self.encoder(x)
        #x_enc = checkpoint.checkpoint(self.encoder, x)


        #print("x_enc", x_enc.shape)
        x_tr = self.swin1(x_enc)
        #print("x_tr", x_tr.shape)

        y = self.decoder(x_tr)
        y = y.view(B, Nlat, Nlon, self.O)



        return y

