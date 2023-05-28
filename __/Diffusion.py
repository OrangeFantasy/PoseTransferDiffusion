import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F


def _betas(T, beta_1: float = 0.0001, beta_T: float = 0.02, cosine_s: float = 0.008, schedule = "linear"):
    betas = None
    if schedule == "linear":
        betas = torch.linspace(beta_1, beta_T, T)
    elif schedule == "quad":
        betas =  torch.linspace(math.sqrt(beta_1), math.sqrt(beta_T), T) ** 2
    elif schedule == "cosine":
        timesteps = torch.arange(T + 1, dtype=torch.float64) / T + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)

    return betas


def probability_mask(probability: float, shape, device) -> Tensor:
    if probability == 1.:
        return torch.ones(shape, dtype=torch.bool, device=device)
    elif probability == 0.:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < probability


class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, T: int, beta_1: float, beta_T: float, 
                 p_mask: float = 0.2, w_s: float = 0.5, w_p: float = 0.5) -> None:
        super().__init__()
        self.model = model
        self.T = T

        self.register_buffer('betas', _betas(T, beta_1, beta_T, schedule="linear"))
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1., 0.], value=1.)[:T]

        # forward diffusion parameters.
        self.p_mask = p_mask

        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1. - alphas_bar))

        # reverse denoising parameters.
        self.w_s = w_s
        self.w_p = w_p

        self.register_buffer("coeff1", torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_sample(self, x_0, t, noise):
        x_t = self.sqrt_alphas_bar.index_select(dim=0, index=t).view(-1, 1, 1, 1) * x_0 + \
            self.sqrt_one_minus_alphas_bar.index_select(dim=0, index=t).view(-1, 1, 1, 1) * noise
        return x_t

    def train_model(self, x_0: Tensor, condition: list[Tensor] | tuple[Tensor]):
        """
        :param x_0: target image.
        :param condition: a list or tuple include source image and target pose.
        :return: mse loss with \eplision and \eplision_{\\theta}
        """
        c_mask = probability_mask(self.p_mask, x_0.shape[0], x_0.device)
        source_image = condition[0] * c_mask
        target_pose = condition[1] * c_mask
        
        assert x_0.shape == source_image.shape == target_pose.shape

        t = torch.randint(self.T, size=[x_0.shape[0]], device=x_0.device)
        eps = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise=eps)

        eps_theta = self.model.forward(x=torch.cat([x_t, target_pose], dim=1), t=t, c=source_image)
        loss = F.mse_loss(eps_theta, eps)
        return loss
    
    @torch.no_grad()
    def p_mean_variance(self, x_t: Tensor, t: Tensor, condition: list[Tensor] | tuple[Tensor]):
        source_image, target_pose = condition
    
        eps_cond = self.model.forward(x=torch.cat([x_t, target_pose], dim=1), t=t, c=source_image)
        eps_style = self.model.forward(x=torch.cat([x_t, torch.zeros_like(target_pose)], dim=1), t=t, c=source_image)
        eps_pose = self.model.forward(x=torch.cat([x_t, target_pose], dim=1), t=t, c=torch.zeros_like(source_image))
        eps = eps_cond - self.w_s * eps_style - self.w_p * eps_pose

        xt_prev_mean = self.coeff1.index_select(0, index=t).view(-1, 1, 1, 1) * x_t - \
            self.coeff2.index_select(0, index=t).view(-1, 1, 1, 1) * eps
        
        var = self.posterior_var.index_select(0, index=t).view(-1, 1, 1, 1)

        return xt_prev_mean, var

    @torch.no_grad()
    def sample(self, x_T: Tensor, condition: list[Tensor] | tuple[Tensor]):
        x_t = x_T
        t = torch.ones([x_T.shape[0]], dtype=torch.int64, device=x_T.device) * self.T
        for time_step in reversed(range(self.T)):
            mean, var = self.p_mean_variance(x_t, t, condition)

            noise = torch.randn_like(x_t) if time_step > 0 else 0
            x_t = mean + torch.sqrt(var) * noise

            t = t - 1
            print(time_step)
        
        x_0 = torch.clip(x_t, -1, 1)
        return x_0



    def forward(self, input):
        pass
