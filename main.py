import sys
import torch

from omegaconf import OmegaConf
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from diffusion.utils import instantiate_from_config


def sys_setting(seed: int, precision: str = "high"):
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


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/diffusion_config.yaml")
    parser.add_argument("--status", type=str, default="train", help="train or test.")

    # Trainer args.
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default="./checkpoints/epoch=29-step=22500.ckpt")
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

    ckpt_callback = ModelCheckpoint(dirpath="./checkpoints/")

    trainer = Trainer.from_argparse_args(opt, callbacks=ckpt_callback)
    if opt.status == "train":
        trainer.fit(model, dataloader)
    elif opt.status == "test":
        trainer.test(model, dataloader)
    