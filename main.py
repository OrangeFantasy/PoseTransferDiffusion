import sys
import torch

from omegaconf import OmegaConf
from argparse import ArgumentParser
from pytorch_lightning import Trainer

from diffusion.utils import instantiate_from_config


def sys_setting(seed: int, precision: str = "medium"):
    torch.set_float32_matmul_precision(precision)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    from torch.backends import cudnn
    cudnn.deterministic = True

    import random
    import numpy
    random.seed(seed)
    numpy.random.seed(seed)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/diffusion_config.yaml")

    # Trainer args.
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--device", type=int, default=1)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()

    sys_setting(seed=opt.seed, precision="medium")
    config = OmegaConf.load(opt.config)

    dataloader = instantiate_from_config(config.data)
    model = instantiate_from_config(config.model).cuda()
    if sys.platform == "linux":
        model = torch.compile(model)
        print("Platform: Linux. Use compiled model.")

    trainer = Trainer.from_argparse_args(opt)
    trainer.fit(model, dataloader)


x = torch.tensor([123])
x.sigmoid().data