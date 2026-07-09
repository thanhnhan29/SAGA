import argparse
import torch
import os
import imageio
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from saga.noise import generate_dual_ar_noise
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--idx", type=int, default=0,
                    help="Start from this dataset sample index (skip all earlier samples)")
parser.add_argument("--max_samples", type=int, default=-1,
                    help="Maximum number of dataset samples to process from idx. -1 means all remaining samples")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--noise_type", type=str, default="dual_ar", choices=["iid", "dual_ar"],
                    help="Noise sampler for T2V: iid for torch.randn baseline, dual_ar for structured noise.")
# Slepian temporal guidance
parser.add_argument("--slepian_guidance_scale", type=float, default=3,
                    help="Slepian HF acceleration guidance scale λ (0 = disabled)")
parser.add_argument("--slepian_nw", type=float, default=1.5,
                    help="Slepian half-bandwidth NW")
parser.add_argument("--slepian_k_cutoff", type=int, default=3,
                    help="Slepian orders k >= k_cutoff treated as high-frequency")
parser.add_argument("--slepian_window", type=int, default=9,
                    help="Number of frames for Slepian acceleration window")
parser.add_argument("--slepian_recurrence_steps", type=int, default=2,
                    help="Forward-backward recurrence steps after Slepian guidance (0=off, 1-2 recommended)")
parser.add_argument("--slepian_loss_type", type=int, default=1, choices=[1, 2],
                    help="1 for acceleration-based slepian loss, 2 for direct latent slepian loss (ablation).")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = False

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
    print("======== Running in few-step inference mode ========")
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()



def antiphase_noise_sampling(
    shape,
    rho=-0.7,
    device="cuda",
    dtype=torch.float32,
):
    """
    Generate temporally anti-correlated noise (AR(1) with rho < 0)
    shape: (B, T, C, H, W)
    """
    B, T, C, H, W = shape
    eps = torch.randn(B, T, C, H, W, device=device, dtype=dtype)
    z = torch.zeros_like(eps)
    z[:, 0] = eps[:, 0]
    scale = (1 - rho**2) ** 0.5
    for t in range(1, T):
        # rho_t = rho * (1 - t / T)
        scale = (1 - rho**2) ** 0.5
        z[:,t] = rho * z[:,t-1] + scale * eps[:,t]
    return z

import math
import torch

def alternating_ar_noise_sampling(
    shape,
    rho_push=-0.9,  # Push coefficient (encourages motion)
    rho_hold=0.9,   # Hold coefficient (preserves consistency)
    device="cuda",
    dtype=torch.float32,
):
    B, T, C, H, W = shape
    eps = torch.randn(B, T, C, H, W, device=device, dtype=dtype)
    z = torch.zeros_like(eps)
    
    # Initialize the first frame.
    z[:, 0] = eps[:, 0]
    
    # Alternate between push and hold coefficients.
    for t in range(1, T):
        # Odd frames use the push coefficient.
        if t % 2 != 0:
            rho_t = rho_push
        # Even frames use the hold coefficient.
        else:
            rho_t = rho_hold
            
        # Preserve unit variance.
        scale_t = math.sqrt(1 - rho_t**2)
        
        # Apply the autoregressive update.
        z[:, t] = rho_t * z[:, t-1] + scale_t * eps[:, t]
        
    return z

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # Resume support: skip samples before the requested start index.
    if idx < args.idx:
        continue
    if args.max_samples > 0 and idx >= args.idx + args.max_samples:
        break

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)
        set_seed(args.seed)
        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
        set_seed(args.seed)
        if args.slepian_guidance_scale > 0.0:
            sampled_noise = generate_dual_ar_noise(
                shape=[args.num_samples, args.num_output_frames - 1, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16
            )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None
        # Keep noise generation reproducible across runs.
        set_seed(args.seed)
        noise_shape = (args.num_samples, args.num_output_frames, 16, 60, 104)
        if args.noise_type == "iid":
            sampled_noise = torch.randn(noise_shape, device=device, dtype=torch.bfloat16)
        elif args.noise_type == "dual_ar":
            sampled_noise = generate_dual_ar_noise(noise_shape, device=device, dtype=torch.bfloat16)
        else:
            raise ValueError(f"Unsupported noise_type: {args.noise_type}")

    # Generate 81 frames
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        low_memory=low_memory,
        slepian_guidance_scale=args.slepian_guidance_scale,
        slepian_nw=args.slepian_nw,
        slepian_k_cutoff=args.slepian_k_cutoff,
        slepian_window=args.slepian_window,
        slepian_recurrence_steps=args.slepian_recurrence_steps,
        slepian_loss_type=getattr(args, 'slepian_loss_type', 1),
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        vid_uint8 = video.to(torch.uint8)  # already 0-255 after 255* above
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            imageio.mimwrite(output_path, vid_uint8[seed_idx].numpy(), fps=16, codec="libx264")
