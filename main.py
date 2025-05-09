import sys
import os
import torch
import pytorch_lightning as pl

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
    parser.add_argument("--status", type=str, default="test", help="train or test.")
    # parser.add_argument("--ckpt_path", type=str, default="checkpoints/epoch=649-step=487500.ckpt")

    # Trainer args.
    parser.add_argument("--max_epochs", type=int, default=900)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--device", type=int, default=1)
    return parser


class CheckpointCallback(ModelCheckpoint):
    def __init__(self, save_ckpt_per_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_ckpt_per_epoch = save_ckpt_per_epoch

    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        
        if self.save_ckpt_per_epoch >= 1 and (trainer.current_epoch + 1) % self.save_ckpt_per_epoch == 0:
            ckpt_name = "ex_ckpt-epoch=" + str(trainer.current_epoch) + "-step=" + str(trainer.global_step) + self.FILE_EXTENSION
            filepath = os.path.join(self.dirpath, ckpt_name)
            trainer.save_checkpoint(filepath, weights_only=True)


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

    ckpt_callback = CheckpointCallback(save_ckpt_per_epoch=50, dirpath="./checkpoints/", every_n_epochs=10)

    trainer = Trainer.from_argparse_args(opt, callbacks=ckpt_callback)
    if opt.status == "train":
        trainer.fit(model, dataloader, ckpt_path=opt.ckpt_path)
    elif opt.status == "test":
        trainer.test(model, dataloader)
    