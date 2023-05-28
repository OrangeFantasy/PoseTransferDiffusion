import sys
import torch

from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from diffusion.utils import instantiate_from_config


def setting(seed: int, precision: str = "medium"):
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


if __name__ == '__main__':
    config = OmegaConf.load("configs/diffusion_config.yaml")

    dataloader = instantiate_from_config(config.data)
    model = instantiate_from_config(config.model).cuda()
    if sys.platform == "linux":
        model = torch.compile(mode)
        print("Platform: Linux. Use compiled model.")

    trainer = Trainer(max_epochs=200, accelerator="gpu", devices=1)
    trainer.fit(model, dataloader)
