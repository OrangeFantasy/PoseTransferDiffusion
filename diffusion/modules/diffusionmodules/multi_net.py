import torch
from torch import nn, Tensor
from torch.nn import functional as F
from abc import abstractmethod
from einops import rearrange

import sys, os
sys.path.append(os.getcwd())
from diffusion.modules.diffusionmodules.unet import Downsample, Upsample
sys.path.pop()


class UNetModel(nn.Module):
    def __init__(self, in_ch: int, 
                 model_ch: int, 
                 out_ch: int, 
                 num_res_blocks: int | list, 
                 ch_mult: list) -> None:
        super().__init__()

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(ch_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(ch_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        self.input_blocks = nn.ModuleList()
        for level, mult in enumerate(ch_mult):
            curr_ch = model_ch * mult
            for _ in range(self.num_res_blocks[level]):
                self.input_blocks.append(nn.Conv2d(curr_ch, curr_ch, kernel_size=1, stride=1, padding=0))

            if level != len(ch_mult) - 1:
                self.input_blocks.append(Downsample(curr_ch))

        self.middle_blocks = nn.ModuleList(
            nn.Conv2d(curr_ch, curr_ch, kernel_size=1, stride=1, padding=1),
            nn.Conv2d(curr_ch, curr_ch, kernel_size=1, stride=1, padding=1),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(ch_mult))[::-1]:
            curr_ch = model_ch * mult
            for i in range(self.num_res_blocks[level] + 1):
                self.input_blocks.append([nn.Conv2d(curr_ch, curr_ch, kernel_size=1, stride=1, padding=0)])

            if level != 0 and i == self.num_res_blocks[level]:
                self.input_blocks.append(Upsample(curr_ch))
                
    def forward(self, x):
        hs = []

        for block in [self.input_blocks, self.middle_blocks, self.output_blocks]:
            f_x = block(x)
            hs.append(x)
        

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.InstanceNorm2d(ch),
            nn.SiLU()
        )

        self.beta = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.gamma = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x, emb, w):
        x = self.in_layer(x) + emb

        beta = self.beta(w)
        gamma = self.gamma(w)

        x = beta + x * (1 + gamma)
        return x


class ResBlockWithAdaptiveInstanceNorm(TimestepBlock):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch)
        )
        self.res = ResBlock(in_ch, out_ch, dropout)
        self.ada_norm = AdaptiveInstanceNorm(out_ch)

        
        
        # Note: next Attention block don't need norm again.

    def forward(self, x, emb, adaptive_map: Tensor = None):
        emb = self.emb_layers(emb)[:, :, None, None]
        x = self.res(x, emb)
        x = self.ada_norm(x, emb, adaptive_map)

        return x


class MappingNetwork(nn.Module):
    def __init__(self, z_ch, w_ch, num_layers) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([nn.Conv2d(z_ch, w_ch, kernel_size=1, stride=1, padding=0)])
        for _ in range(num_layers):
            self.blocks.append(nn.Sequential(
                nn.SiLU(),
                nn.Conv2d(w_ch, w_ch, kernel_size=1, stride=1, padding=0)
            ))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
