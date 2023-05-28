import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange

from diffusion.modules.diffusionmodules.utils import zero_module

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_out = None, mult: int = 4, glu: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_in * mult
        if dim_out is None:
            dim_out = dim_in
        
        self.block = nn.Sequential(
            GEGLU(dim_in, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )
    
    def forward(self, x) -> Tensor:
        return self.block(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, n_heads: int = 8, dim_head: int = 64, context_dim: int = None, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * n_heads
        if context_dim is None:
            context_dim = query_dim
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: Tensor, context: Tensor = None) -> Tensor:
        if context is None:
            context = x

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda _it : rearrange(_it, "b n (h d) -> (b h) n d", h=self.n_heads), [q, k, v])

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
        sim = torch.softmax(sim, dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", sim, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.n_heads)
        out = self.to_out(out)

        return out


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dim_head: int, context_dim: int = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, n_heads, dim_head, None, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, n_heads, dim_head, context_dim, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, x: Tensor, context: Tensor = None):
        x = self.attn1(self.norm1(x), None) + x
        x = self.attn2(self.norm2(x), context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, in_ch: int, n_heads: int, dim_head: int, depth:int = 1, context_dim: int = None, dropout: float = 0.0) -> None:
        super().__init__()

        if not isinstance(context_dim, list):
            context_dim = [context_dim]
        inner_dim = n_heads * dim_head

        self.norm = nn.GroupNorm(32, in_ch, eps=1e-6)
        self.proj_in = nn.Conv2d(in_ch, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, dim_head, dropout=dropout, context_dim=context_dim[i])
             for i in range(depth)]
        )
        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_ch, kernel_size=1, stride=1, padding=0))

    def forward(self, x: Tensor, context: Tensor = None) -> Tensor:
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]

        b, c, h, w = x.shape

        x_ = self.norm(x)
        x_ = self.proj_in(x_)
        x_ = rearrange(x_, 'b c h w -> b (h w) c')

        for i, block in enumerate(self.transformer_blocks):
            x_ = block(x_, context=context[i])

        x_ = rearrange(x_, 'b (h w) c -> b c h w', h=h, w=w)
        x_ = self.proj_out(x_)

        return x + x_


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        if context_dim is None:
            context_dim = query_dim

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op = None

    def forward(self, x, context=None, mask=None):
        if context is None:
            context = x

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if mask is not None:
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)



if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # block = CrossAttention(128, 8, 64, 128)
    # x = torch.rand([1, 128, 128])
    # context = torch.rand([1, 128, 128])
    # out = block.forward(x, context)
    # print(out.shape)

    # block = BasicTransformerBlock(128, 8, 64, 128)
    # x = torch.rand([1, 128, 128])
    # context = torch.rand([1, 128, 128])
    # out = block.forward(x, context)
    # print(out.shape)

    block = SpatialTransformer(128, 16, 64, depth=1, context_dim=1024)
    x = torch.rand([1, 128, 32, 32])
    context = torch.rand([1, 4, 1024])
    out = block.forward(x, context)
    print(out[:, 0, :, 0])

    pass
