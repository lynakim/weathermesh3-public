import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)

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

class Attention(nn.Module):
    """
    If 'context' is None, does self-attention on x.
    Otherwise, does cross-attention where x attends to 'context'.
    'mask' should be a boolean mask of shape [batch, ctx_len],
      where True entries indicate tokens to be *ignored*.
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=True)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x.shape = [B, Nx, D]
        # context.shape = [B, Nc, D] (if cross-attending)
        # mask.shape = [B, Nc] with True for tokens to ignore (if cross-attending)
        if context is None:
            context = x

        b, nx, _ = x.shape
        _, nc, _ = context.shape
        h = self.heads

        # Compute Q from x; K and V from context
        q = self.to_q(x).view(b, nx, h, self.dim_head)
        kv = self.to_kv(context).view(b, nc, 2, h, self.dim_head)
        k, v = kv[:, :, 0], kv[:, :, 1]

        # Rearrange for scaled_dot_product_attention
        # q, k, v => [B, head, seq_len, dim_head]
        q = q.permute(0, 2, 1, 3)  # [b, h, nx, dim_head]
        k = k.permute(0, 2, 1, 3)  # [b, h, nc, dim_head]
        v = v.permute(0, 2, 1, 3)  # [b, h, nc, dim_head]

        # Convert a [B, Nc] boolean mask to [B, 1, 1, Nc] for broadcast
        # scaled_dot_product_attention treats True as "ignore"
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :]  # [b,1,1,nc]

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,       # True => ignore that token
            dropout_p=0.0,            # Adjust if needed
            is_causal=False
        )
        # out => [b, h, nx, dim_head], reshape back
        out = out.permute(0, 2, 1, 3).contiguous().view(b, nx, -1)
        return self.to_out(out)

class CrossTransformer(nn.Module):
    """
    A transformer that:
      1) Cross-attends x to ctx
      2) Followed by a self-attention layer on x
      3) Repeats above for 'depth' layers
      4) Accepts optional 'ctx_mask' to zero out some context tokens
      5) Uses PyTorch 2.0+ scaled_dot_product_attention for flash attention
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        dim_head: int = 64,
        mlp_dim: Optional[int] = None,
        checkpoint_fn: Optional[Callable] = None
    ):
        super().__init__()
        mlp_dim = mlp_dim or (dim * 4)
        self.checkpoint_fn = checkpoint_fn or (lambda f, *args, **kwargs: f(*args, **kwargs))

        heads = dim // dim_head
        self.layers = nn.ModuleList()
        for _ in range(depth):
            # Each layer: Cross-Attn -> FF -> Self-Attn -> FF
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head)),  # cross-attn
                PreNorm(dim, Attention(dim, heads, dim_head)),  # self-attn
                PreNorm(dim, FeedForward(dim, mlp_dim)),
            ]))

    def forward(
        self,
        x: torch.Tensor,           # [B, Nx, D]
        ctx: torch.Tensor,         # [B, Nc, D]
        ctx_mask: Optional[torch.Tensor] = None  # [B, Nc], True => ignore
    ) -> torch.Tensor:
        for (cross_attn, self_attn, self_ff) in self.layers:
            # Cross-attention
            x = self.checkpoint_fn(cross_attn, x, context=ctx, mask=ctx_mask) + x

            # Self-attention
            x = self.checkpoint_fn(self_attn, x) + x
            x = self.checkpoint_fn(self_ff, x) + x

        return x


if __name__ == "__main__":
    # Example:
    model = CrossTransformer(dim=128, depth=2)
    x = torch.randn(4, 10, 128)       # batch=4, seq_len=10, dim=128
    ctx = torch.randn(4, 12, 128)     # batch=4, context_len=12, dim=128
    ctx_mask = torch.zeros(4, 12); ctx_mask[:,-1] = 1  # last token is ignored
    out = model(x, ctx, ctx_mask)     # [4, 10, 128]