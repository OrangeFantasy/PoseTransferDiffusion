import torch
from torch import nn, Tensor
from torch.nn import functional as F
from abc import abstractmethod

import sys, os
sys.path.append(os.getcwd())
from diffusion.modules.attention import SpatialTransformer, SpatialAttentionBlock
from diffusion.modules.diffusionmodules.condition_blocks import MappingNetwork, MultiScaleEncoder
from diffusion.modules.diffusionmodules.utils import timestep_embedding, zero_module
sys.path.pop()


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, *args):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that support it as an extra input.
    """

    def forward(self, x: Tensor, emb: Tensor, context: Tensor = None, *args) -> Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, *args)
            elif isinstance(layer, SpatialAttentionBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x
    

class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = None):
        super().__init__()

        if out_ch is None:
            out_ch = in_ch
        
        self.op = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.op(x)
        return x
    

class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = None):
        super().__init__()

        if out_ch is None:
            out_ch = in_ch
        
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.conv(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, ch, w_ch, emb_dim: int) -> None:
        super().__init__()

        self.norm_x = nn.InstanceNorm2d(ch, affine=False)

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, w_ch)
        )

        self.shared = nn.Sequential(
            nn.InstanceNorm2d(w_ch, affine=False),
            nn.SiLU()
        )
        if ch != w_ch:
            self.shared.add_module("skip_connection", nn.Conv2d(w_ch, ch, kernel_size=1, stride=1, padding=0))
        
        self.beta = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.gamma = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)


    def forward(self, x, w, emb):
        x = self.norm_x(x)
        w = w + self.emb_layers(emb)[:, :, None, None]

        w = self.shared(w)
        beta = self.beta(w)
        gamma = self.gamma(w)

        x = x * (1 + gamma) + beta
        return x


class AdaptiveResBlock(TimestepBlock):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.in_norm = AdaptiveInstanceNorm2d(ch=in_ch, w_ch=in_ch, emb_dim=emb_dim)
        self.in_layers = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

        self.out_norm = AdaptiveInstanceNorm2d(ch=out_ch, w_ch=in_ch, emb_dim=emb_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1))
        )

        if in_ch == out_ch:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, emb, *args):
        w = args[0]

        h = self.in_layers(self.in_norm(x, w, emb))
        h = self.out_layers(self.out_norm(h, w, emb))

        x = self.skip_connection(x)
        return x + h


class ResBlock(TimestepBlock):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch)
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1))
        )

        if in_ch == out_ch:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, emb, *args):
        h = self.in_layers(x)
        h = h + self.emb_layers(emb)[:, :, None, None]
        h = self.out_layers(h)

        x = self.skip_connection(x)
        return x + h


class UNetModel(nn.Module):
    def __init__(self, in_ch: int, 
                 model_ch: int, 
                 out_ch: int, 
                 num_res_blocks: int | list, 
                 attn_resolutions: list, 
                 ch_mult: list,
                 n_heads: int       = -1, 
                 dim_head: int      = -1,
                 context_dim: int   = None,
                 context_ch: int    = None, 
                 dropout: float     = 0.) -> None:
        super().__init__()

        assert n_heads != -1 or dim_head != -1, "Either num_heads or num_head_channels has to be set"

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(ch_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(ch_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) \
                                 or as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        
        self.model_ch = model_ch

        # Time Embedding.
        temb_dim = model_ch * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_ch, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim)
        )

        # Condition.
        self.condition_mapping = MappingNetwork(context_ch, context_dim, num_layers=8)
        self.multi_scale_encoder = MultiScaleEncoder(context_dim, model_ch, self.num_res_blocks, ch_mult)

        # Input Blocks.
        curr_ch = model_ch
        input_chs = [model_ch]
        down_resolution = 1

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_ch, model_ch, kernel_size=3, stride=1, padding=1))
        ])
        for level, mult in enumerate(ch_mult):
            _ch = model_ch * mult
            for _ in range(self.num_res_blocks[level]):
                _blocks = [AdaptiveResBlock(curr_ch, _ch, temb_dim, dropout)]
                curr_ch = _ch

                if down_resolution in attn_resolutions:
                    if dim_head == -1:
                        _dim_head = curr_ch // n_heads
                    else:
                        n_heads = curr_ch // dim_head
                        _dim_head = dim_head
                    _blocks.append(SpatialAttentionBlock(curr_ch, n_heads, _dim_head, depth=1, context_dim=context_dim, dropout=dropout))
                
                self.input_blocks.append(TimestepEmbedSequential(*_blocks))
                input_chs.append(curr_ch)

            if level != len(ch_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(curr_ch)))
                input_chs.append(curr_ch)
                down_resolution *= 2

        # Middle Blocks.
        if dim_head == -1:
            _dim_head = curr_ch // n_heads
        else:
            n_heads = curr_ch // dim_head
            _dim_head = dim_head
        self.middle_blocks = TimestepEmbedSequential(
            AdaptiveResBlock(curr_ch, curr_ch, temb_dim, dropout),
            SpatialAttentionBlock(curr_ch, n_heads, _dim_head, depth=1, context_dim=context_dim, dropout=dropout),
            ResBlock(curr_ch, curr_ch, temb_dim, dropout),
        )

        # Output Blocks.
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(ch_mult))[::-1]:
            _ch = model_ch * mult
            for i in range(self.num_res_blocks[level] + 1):
                _blocks = [ResBlock(curr_ch + input_chs.pop(), _ch, temb_dim, dropout)]
                curr_ch = _ch

                if down_resolution in attn_resolutions:
                    if dim_head == -1:
                        _dim_head = curr_ch // n_heads
                    else:
                        n_heads = curr_ch // dim_head
                        _dim_head = dim_head
                    _blocks.append(SpatialAttentionBlock(curr_ch, n_heads, _dim_head, depth=1, context_dim=context_dim, dropout=dropout))

                if level != 0 and i == self.num_res_blocks[level]:
                    _blocks.append(Upsample(curr_ch))
                    down_resolution //= 2
                
                self.output_blocks.append(TimestepEmbedSequential(*_blocks))
        
        self.out = nn.Sequential(
            nn.GroupNorm(32, curr_ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(curr_ch, out_ch, kernel_size=3, stride=1, padding=1))
        )
    
    def forward(self, x: Tensor, timesteps: Tensor, context: Tensor = None) -> Tensor:
        hs = []
        temb = timestep_embedding(timesteps, self.model_ch)
        temb = self.time_embed(temb)
        
        context = self.condition_mapping(context)
        multi_scale_context = self.multi_scale_encoder(context)

        h = x
        for idx, module in enumerate(self.input_blocks):
            h = module(h, temb, context, multi_scale_context[idx])
            hs.append(h)
        h = self.middle_blocks(h, temb, context, multi_scale_context[-1])
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, temb, context)

        h = self.out(h)
        return h


if __name__ == "__main__":
    model = UNetModel(8, 128, 4, num_res_blocks=2, attn_resolutions=[], 
                      ch_mult=[1, 2, 4, 4], dim_head=64, context_dim=128, context_ch=4).cuda()
    x = torch.rand([1, 8, 32, 32]).cuda()  # [b, c, h, w]
    t = torch.randint(0, 1000, [1]).cuda()  # [b]
    context = torch.rand([1, 4, 32, 32]).cuda()  #
    out = model.forward(x, t, context)
    print(out.shape)
