import torch
from torch import nn, Tensor
from typing import Optional, Tuple
from natten.utils import check_all_args
from natten.types import CausalArg2DTypeOrDed, Dimension2DTypeOrDed
from natten.functional import FusedNeighborhoodAttention2D
from model_latlon.primatives2d import call_checkpointed

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

class Natten2DTransformerBlock(nn.Module):
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

class SlideLayers2D(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, harebrained=False, checkpoint_type="matepoint"):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.checkpoint_type = checkpoint_type
        assert harebrained == False, "Todo for Joan"
        # Implementing harebrained here requires a bit of work

        mlist = []
        for _ in range(depth):
            tb_block = Natten2DTransformerBlock(
                dim,
                num_heads,
                window_size=window_size,
            )
            mlist.append(tb_block)
        self.blocks = nn.ModuleList(mlist)

    def forward(self, x):

        for _, blk in enumerate(self.blocks):
            x = call_checkpointed(blk, x, checkpoint_type=self.checkpoint_type)

        return x