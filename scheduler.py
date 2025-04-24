import numpy as np

import torch

import torch.nn.functional as F

from tqdm import tqdm

def extract_into_tensor(arr, timesteps, broadcast_shape):

    res = torch.from_numpy(arr).to(torch.float32).to(device=timesteps.device)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)

class Scheduler:

    def __init__(self, denoise_model, denoise_steps, beta_start=1e-4, beta_end=0.02):

        self.model = denoise_model

        betas = np.array(
            np.linspace(beta_start, beta_end, denoise_steps),
            dtype=np.float64
        )

        self.denoise_steps = denoise_steps

        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        alphas = 1.0 - betas

        self.sqrt_alphas = np.sqrt(alphas)
        self.one_minus_alphas = 1.0 - alphas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

    def q_sample(self, x0, t, noise):

        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def training_losses(self, x, t):

        noise = torch.randn_like(x)
        x_t = self.q_sample(x, t, noise)

        predict_noise = self.model(x_t, t)

        return F.mse_loss(predict_noise, noise)


    @torch.no_grad()
    def ddpm(self, sample_shape, device):

        x = torch.randn(*sample_shape, device=device)

        for t in tqdm(reversed(range(0, self.denoise_steps)), total=self.denoise_steps):

            t = torch.tensor([t], device=device).repeat(sample_shape[0])
            t_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))

            eps = self.model(x, t)

            x = x - (
                    extract_into_tensor(self.one_minus_alphas, t, x.shape) * eps
                    / extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            )

            x = x / extract_into_tensor(self.sqrt_alphas, t, x.shape)

            sigma = torch.sqrt(
                extract_into_tensor(self.one_minus_alphas, t, x.shape)
                * (1.0 - extract_into_tensor(self.alphas_cumprod_prev, t, x.shape))
                / (1.0 - extract_into_tensor(self.alphas_cumprod, t, x.shape))
            )

            x = x + sigma * torch.randn_like(x) * t_mask

            x = x.clip(-1, 1)

        return x


    @torch.no_grad()
    def ddim(self, sample_shape, device, eta=0.0, sub_sequence_step=25):

        x = torch.randn(*sample_shape, device=device)

        t_seq = list(range(self.denoise_steps - 1, -1, -sub_sequence_step))
        for i in tqdm(range(len(t_seq)), total=len(t_seq)):

            t = t_seq[i]
            s = 0 if i == len(t_seq) - 1 else t_seq[i + 1]

            t_tensor = torch.tensor([t], device=device).repeat(sample_shape[0])
            s_tensor = torch.tensor([s], device=device).repeat(sample_shape[0])

            eps = self.model(x, t_tensor)

            alpha_bar_t = extract_into_tensor(self.alphas_cumprod, t_tensor, x.shape)
            alpha_bar_s = extract_into_tensor(self.alphas_cumprod, s_tensor, x.shape)

            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

            sigma = 0.0
            if eta > 0.0 and s > 0:
                sigma = eta * torch.sqrt(
                    (1 - alpha_bar_s) / (1 - alpha_bar_t) *
                    (1 - alpha_bar_t / alpha_bar_s)
                )

            x = torch.sqrt(alpha_bar_s) * x0_pred + torch.sqrt(1 - alpha_bar_s - sigma ** 2) * eps

            if eta > 0.0 and s > 0:
                x = x + sigma * torch.randn_like(x)

            x = x.clip(-1, 1)

        return x
