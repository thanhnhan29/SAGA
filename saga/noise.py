import math

import torch


def generate_iid_noise(shape, device="cuda", dtype=torch.float32):
    return torch.randn(shape, device=device, dtype=dtype)


def generate_dual_ar_noise(
        shape,
        num_frames=None,
        rho_lf=0.9,
        rho_hf=-0.9,
        theta=math.pi / 4,
        device="cuda",
        dtype=torch.float32,
):
    """
    Dual-AR temporal noise generator.

    The sampler mixes a low-frequency AR(1) stream with a high-frequency
    anti-correlated AR(1) stream, returning noise in (B, T, C, H, W) layout.
    """
    if len(shape) == 5:
        bsz, t_from_shape, ch, h, w = shape
        total_frames = t_from_shape if num_frames is None else num_frames
        frame_shape = (bsz, ch, h, w)
    elif len(shape) == 4:
        if num_frames is None:
            raise ValueError("num_frames must be provided when shape is (B, C, H, W)")
        bsz, ch, h, w = shape
        total_frames = num_frames
        frame_shape = (bsz, ch, h, w)
    else:
        raise ValueError("shape must be (B, T, C, H, W) or (B, C, H, W)")

    alpha, beta = math.cos(theta), math.sin(theta)

    z_lf = torch.randn(frame_shape, device=device, dtype=dtype)
    z_hf = torch.randn(frame_shape, device=device, dtype=dtype)
    video_noise = [alpha * z_lf + beta * z_hf]

    for _ in range(1, total_frames):
        eps_lf = torch.randn(frame_shape, device=device, dtype=dtype)
        eps_hf = torch.randn(frame_shape, device=device, dtype=dtype)

        z_lf = rho_lf * z_lf + math.sqrt(1 - rho_lf ** 2) * eps_lf
        z_hf = rho_hf * z_hf + math.sqrt(1 - rho_hf ** 2) * eps_hf

        video_noise.append(alpha * z_lf + beta * z_hf)

    return torch.stack(video_noise, dim=1)

