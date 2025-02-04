# (C) Claude, ChatGPT o1
import torch
import torch.nn as nn
from typing import Optional, Callable

# Brought to you by Joan's Ministry of Developer-friendly Code

class ConvenientTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        dim_head: int = 64,
        mlp_dim: Optional[int] = None,
        checkpoint_fn: Optional[Callable] = None,
        make_tokens: int = 0,
        batch_first: bool = True  # <--- new argument
    ):
        super().__init__()
        self.make_tokens = make_tokens
        self.batch_first = batch_first  # <--- store it
        mlp_dim = mlp_dim or dim * 4
        self.checkpoint_fn = checkpoint_fn or (lambda f, *args: f(*args))
        heads = dim // dim_head
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim))
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects either:
          - If batch_first == True:  x is [batch, seq, dim] or possibly [batch, dim] if 2D
          - If batch_first == False: x is [seq, batch, dim] or possibly [seq, batch] if 2D

        We transpose only if batch_first is False, so submodules always see [batch, seq, ...].
        """
        # If not batch_first, transpose from [seq, batch, ...] -> [batch, seq, ...]
        if not self.batch_first:
            x = x.transpose(0, 1)

        # Optional token creation logic
        if self.make_tokens > 0:
            # e.g. if x was [B, D], treat it as [B, make_tokens, ...]
            B, D = x.shape
            # Make sure the dimension is divisible
            assert D % (self.make_tokens * x.shape[-1]) == 0, \
                'Input dimension must be divisible by make_tokens * dim'
            x = x.view(B, self.make_tokens, -1)

        # Pass through the transformer layers
        for attn, ff in self.layers:
            x = self.checkpoint_fn(attn, x) + x
            x = self.checkpoint_fn(ff, x) + x

        # If not batch_first, transpose back to [seq, batch, ...]
        if not self.batch_first:
            x = x.transpose(0, 1)

        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        
        # Single QKV projection
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is assumed [batch, seq, dim]
        b, n, _ = x.shape
        h = self.heads
        
        # One matrix multiply, then reshape and chunk
        qkv = self.to_qkv(x).view(b, n, 3, h, self.dim_head)
        qkv = qkv.permute(0, 2, 3, 1, 4)
        q, k, v = qkv.unbind(dim=1)  # Efficiently split along dim 2

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=False
        )
        out = out.permute(0, 2, 1, 3)
        
        out = out.view(b, n, -1)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)
