import sys, os
sys.path.append(os.getcwd())

import numpy as np
import torch
import pytorch_lightning as pl
from torch import Tensor
from torch.nn import functional as F
from functools import partial

from diffusion.modules.diffusionmodules.utils import make_beta_schedule, extract_into_tensor
from diffusion.utils import instantiate_from_config
from diffusion.modules.distributions import DiagonalGaussianDistribution


class DDPM(pl.LightningModule):
    def __init__(self, 
                 unet_config, 
                 timesteps: int     = 1000, 
                 lr: float          = 1e-4, 
                 beta_schedule: str = "linear", 
                 beta_1: float      = 1e-4, 
                 beta_T : float     = 2e-2, 
                 cosine_s: float    = 8e-3,
                 loss_type: str     = "l2",
                 v_posterior: float = 0.0,
                 parameterization: str = "eps",
                 scheduler_config   = None,
                 ckpt_path: str     = None) -> None:
        super().__init__()

        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization

        self.learning_rate = lr
        self.scheduler_config = scheduler_config
        self.loss_type = loss_type

        self.model = DiffusionWrapper(unet_config)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.v_posterior = v_posterior
        self.register_schedule(beta_schedule, timesteps, beta_1, beta_T, cosine_s)


    def register_schedule(self, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = int(betas.shape[0])
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer("posterior_mean_coef2", to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        ckpt = torch.load(path, map_location="cpu")
        keys = list(ckpt.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del ckpt[k]
        self.load_state_dict(ckpt, strict=False)
        pass

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor = None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)    
    
    def q_posterior(self, x_start: Tensor, x_t: Tensor, t: Tensor):
        """
        Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        """ 
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def get_loss(self, pred: Tensor, target: Tensor, mean: bool = True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = F.mse_loss(target, pred)
            else:
                loss = F.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss
    
    def p_losses(self, x_start: Tensor, t: Tensor, noise: Tensor = None):
        pass
    
    def forward(self, x: Tensor, *args, **kwargs):
        pass
    
    def shared_step(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False,  rank_zero_only=True)

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False,  rank_zero_only=True)

        return loss

    def configure_optimizers(self):
        pass


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, model_config) -> None:
        super().__init__()
        self.unet_model = instantiate_from_config(model_config)

    def forward(self, x, t, c_concat, c_crossattn):
        # Note: conditioning key is "hybrid".
        xc = torch.cat([x, c_concat], dim=1)
        eps_theta = self.unet_model.forward(xc, t, c_crossattn)
        return eps_theta


class PoseTransferDiffusion(DDPM):
    def __init__(self, 
                 first_stage_config, 
                 cond_stage_config  = None, 
                 scale_factor       = 1.0,
                 *args, 
                 **kwargs) -> None:    
        super().__init__(*args, **kwargs)

        self.scale_factor = scale_factor

        # vae.
        self.first_stage_model = self.instantiate_first_stage(first_stage_config)
        # self.cond_stage_model = self.instantiate_cond_stage(cond_stage_config)
    
    def instantiate_first_stage(self, config) -> pl.LightningModule:
        model = instantiate_from_config(config).eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def instantiate_cond_stage(self, config) -> pl.LightningModule:
        pass
    
    @torch.no_grad()
    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
    
    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)
    
    @torch.no_grad()
    def decode_first_stage(self, z):
        return self.first_stage_model.decode(z)

    def p_losses(self, x_start, t, condition, noise: Tensor = None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        target_pose, source_image = condition
        model_out = self.model.forward(x_t, t, target_pose, source_image)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        loss_simple = self.get_loss(model_out, noise, mean=False).mean(dim=[1, 2, 3])
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        return loss_simple, loss_dict
    
    def forward(self, x, c, *args, **kwargs):
        # Note: condition includes target pose and source image, so it's a list or tuple.
        t = torch.randint(0, self.num_timesteps, size=[x.shape[0]], device=self.device).long()
        loss, loss_dict = self.p_losses(x, t, c)
        return loss, loss_dict
    
    def get_input(self, batch):
        # inputs: batch includes source image, source pose, target image and target pose.
        # return: a target image latent code, a list with concat condition and cross condition.
        src_image, src_pose, tgt_image, tgt_pose = batch

        z = self.get_first_stage_encoding(self.encode_first_stage(src_image))
        c_cross = self.get_first_stage_encoding(self.encode_first_stage(tgt_image))
        c_concat = self.get_first_stage_encoding(self.encode_first_stage(tgt_pose))

        # z_ = self.decode_first_stage(z)
        # c_s = self.decode_first_stage(c_cross)
        # c_p = self.decode_first_stage(c_concat)
        # save_image(torch.cat([z_, c_p, c_s], dim=-1), "decode.png", normalize=True)
        return z, [c_concat, c_cross]

    def shared_step(self, batch):
        z, c = self.get_input(batch)
        loss, loss_dict = self.forward(z, c)
        return loss, loss_dict

    def configure_optimizers(self):    
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        if self.scheduler_config is not None:
            self.scheduler_config["optimizer"] = optimizer
            scheduler = instantiate_from_config(self.scheduler_config)
            return [optimizer, scheduler]
        
        return optimizer


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("Configs/DiffusionConfig.yaml")
    model = instantiate_from_config(config.model).cuda()

    source_image = torch.rand([1, 3, 64, 64]).cuda()
    target_image = torch.rand([1, 3, 64, 64]).cuda()
    target_pose = torch.rand([1, 18, 64, 64]).cuda()

    source_image_enc = torch.rand([1, 128, 1024]).cuda()

    model.forward(x=target_image, condition=[target_pose, source_image_enc])
