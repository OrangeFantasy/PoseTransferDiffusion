import torch
from torch import nn, Tensor


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = None):
        super().__init__()

        if out_ch is None:
            out_ch = in_ch
        
        self.op = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.op(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

        if in_ch == out_ch:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)

        x = self.skip_connection(x)
        return x + h


class MultiScaleEncoder(nn.Module):
    def __init__(self, in_ch: int, 
                 model_ch: int, 
                 num_res_blocks: int | list, 
                 attn_resolutions: list, 
                 ch_mult: list,
                 n_heads: int = -1, 
                 dim_head: int = -1) -> None:
        super().__init__()

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(ch_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(ch_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        # Input Blocks.
        curr_ch = model_ch
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_ch, model_ch, kernel_size=3, stride=1, padding=1)
        ])
        for level, mult in enumerate(ch_mult):
            _ch = model_ch * mult
            for _ in range(self.num_res_blocks[level]):
                _blocks = [ResBlock(curr_ch, _ch)]
                curr_ch = _ch

                self.input_blocks.append(*_blocks)

            if level != len(ch_mult) - 1:
                self.input_blocks.append(Downsample(curr_ch))

    def forward(self, x: Tensor) -> Tensor:
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        
        return hs


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


if __name__ == "__main__":
    condition = torch.rand([1, 4, 32, 32]).cuda()

    mapping = MappingNetwork(4, 128, 8).cuda()
    condition = mapping.forward(condition)

    model = MultiScaleEncoder(128, 128, num_res_blocks=2, attn_resolutions=[], ch_mult=[1, 2, 4, 4], dim_head=64).cuda()
    out = model.forward(condition)
    print(out.shape)