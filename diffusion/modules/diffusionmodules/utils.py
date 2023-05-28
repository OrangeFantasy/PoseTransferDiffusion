import math
import numpy as np
import torch
from torch import nn, Tensor


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "squaredcos_cap_v2":  # used for karlo prior
        # return early
        return betas_for_alpha_bar(
            n_timestep,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def extract_into_tensor(arr: Tensor, timesteps: Tensor, broadcast_shape):
    b, *_ = timesteps.shape
    out = arr.gather(-1, timesteps)
    return out.reshape(b, *((1,) * (len(broadcast_shape) - 1)))


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(timesteps.device)
    emb = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def zero_module(module: nn.Module):
    for param in module.parameters():
        param.detach().zero_()
    return module


def scale_module(module: nn.Module):
    for param in module.parameters():
        param.detach().zero_()
    return module


if __name__ == "__main__":
    timesteps = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    dim = 128
    emb = timestep_embedding(timesteps, dim)
    print(emb.shape)