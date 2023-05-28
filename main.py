from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from diffusion.utils import instantiate_from_config


if __name__ == '__main__':
    config = OmegaConf.load("configs/diffusion_config.yaml")

    dataloader = instantiate_from_config(config.data)
    model = instantiate_from_config(config.model).cuda()

    trainer = Trainer(max_epochs=200, accelerator="gpu", devices=1)
    trainer.fit(model, dataloader)

