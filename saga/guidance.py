import functools
from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal.windows import dpss


@dataclass
class SAGAConfig:
    guidance_scale: float = 0.0
    nw: float = 1.5
    k_cutoff: int = 3
    window: int = 9
    loss_type: int = 1


@functools.lru_cache(maxsize=128)
def get_cached_dpss(length: int, nw: float):
    return np.atleast_2d(dpss(length, nw, length)).copy()


class SAGAGuidance(torch.nn.Module):
    """
    Inference-time SAGA guidance.

    This module has no trainable parameters. It computes a high-frequency
    temporal acceleration loss on predicted clean latents and nudges the
    prediction along the normalized negative gradient.
    """

    def __init__(self, config: SAGAConfig):
        super().__init__()
        self.config = config

    def apply(self, denoised_pred: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.config.guidance_scale <= 0.0 or context.shape[1] == 0:
            return denoised_pred

        with torch.enable_grad():
            z0_in = denoised_pred.detach().requires_grad_(True)
            if self.config.loss_type == 1:
                loss = self._compute_slepian_loss(
                    denoised_pred=z0_in,
                    context=context,
                    nw=self.config.nw,
                    k_cutoff=self.config.k_cutoff,
                    window_size=self.config.window,
                )
            else:
                loss = self._compute_fft_loss(
                    denoised_pred=z0_in,
                    context=context,
                    k_cutoff=self.config.k_cutoff,
                    window_size=self.config.window,
                )

            if not loss.requires_grad:
                grad_z0 = torch.zeros_like(z0_in)
            else:
                grad_z0 = torch.autograd.grad(loss, z0_in)[0]

        if not torch.isfinite(grad_z0).all():
            grad_z0 = torch.nan_to_num(grad_z0, nan=0.0, posinf=0.0, neginf=0.0)

        grad_norm = grad_z0.view(grad_z0.shape[0], -1).norm(dim=1, keepdim=True)
        grad_norm = grad_norm.view(-1, 1, 1, 1, 1) + 1e-6
        grad_z0 = grad_z0 / grad_norm

        return denoised_pred - self.config.guidance_scale * grad_z0

    @staticmethod
    def _compute_slepian_loss(
        denoised_pred: torch.Tensor,
        context: torch.Tensor,
        nw: float = 1.5,
        k_cutoff: int = 3,
        window_size: int = 9,
    ) -> torch.Tensor:
        _, f_block, _, _, _ = denoised_pred.shape
        device = denoised_pred.device

        full_seq = torch.cat([context, denoised_pred], dim=1)
        accel = full_seq[:, 2:] - 2 * full_seq[:, 1:-1] + full_seq[:, :-2]
        t_acc = accel.shape[1]
        win_acc = window_size - 2

        if t_acc < win_acc:
            win_acc = t_acc

        if win_acc <= 2 * nw or k_cutoff >= win_acc:
            return torch.tensor(0.0, device=device, dtype=denoised_pred.dtype)

        accel_wins = accel.unfold(1, win_acc, 1)
        num_windows = accel_wins.shape[1]
        take_windows = min(f_block, num_windows)
        accel_wins = accel_wins[:, -take_windows:]

        basis_np = get_cached_dpss(win_acc, nw)
        basis = torch.from_numpy(basis_np).to(device=device, dtype=accel_wins.dtype)
        beta = torch.einsum("kt, ...t -> ...k", basis, accel_wins)
        beta_hf = beta[..., k_cutoff:]

        if beta_hf.shape[-1] == 0:
            return torch.tensor(0.0, device=device, dtype=denoised_pred.dtype)

        return (beta_hf ** 2).sum()

    @staticmethod
    def _compute_fft_loss(
        denoised_pred: torch.Tensor,
        context: torch.Tensor,
        k_cutoff: int = 3,
        window_size: int = 9,
        norm: str = "ortho",
    ) -> torch.Tensor:
        _, f_block, _, _, _ = denoised_pred.shape
        device = denoised_pred.device
        dtype = denoised_pred.dtype

        full_seq = torch.cat([context, denoised_pred], dim=1)
        accel = full_seq[:, 2:] - 2 * full_seq[:, 1:-1] + full_seq[:, :-2]
        t_acc = accel.shape[1]
        win_acc = window_size - 2

        if t_acc < win_acc:
            win_acc = t_acc

        if win_acc <= k_cutoff + 1:
            return torch.tensor(0.0, device=device, dtype=dtype)

        accel_wins = accel.unfold(1, win_acc, 1)
        num_windows = accel_wins.shape[1]
        take_windows = min(f_block, num_windows)
        accel_wins = accel_wins[:, -take_windows:]

        fft_coeff = torch.fft.rfft(accel_wins.float(), dim=-1, norm=norm)
        hf_coeff = fft_coeff[..., k_cutoff:]

        if hf_coeff.shape[-1] == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)

        return hf_coeff.abs().pow(2).sum().to(dtype)

