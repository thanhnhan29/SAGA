from causvid.models.wan.causal_inference import InferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
import os
import numpy as np
import random

from saga.noise import generate_dual_ar_noise

def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)



parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--prompt_file_path", type=str)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--slepian_guidance_scale", type=float, default=0.0)
parser.add_argument("--slepian_nw", type=float, default=3.0)
parser.add_argument("--slepian_k_cutoff", type=int, default=2)
parser.add_argument("--slepian_window", type=int, default=5)

args = parser.parse_args()

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

pipeline = InferencePipeline(config, device="cuda")
pipeline.to(device="cuda", dtype=torch.bfloat16)

state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")[
    'generator']

pipeline.generator.load_state_dict(
    state_dict, strict=True
)

dataset = TextDataset(args.prompt_file_path)

sampled_noise = torch.randn(
    [1, 21, 16, 60, 104], device="cuda", dtype=torch.bfloat16
)
if args.slepian_guidance_scale > 0:
    set_seed(args.seed)
    sampled_noise= generate_dual_ar_noise((args.num_samples, 21, 16, 60, 104),
        # rho=1,
        device="cuda", dtype=torch.bfloat16,
    )

os.makedirs(args.output_folder, exist_ok=True)

for prompt_index in tqdm(range(len(dataset))):
    prompts = [dataset[prompt_index]]

    video = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        slepian_guidance_scale=args.slepian_guidance_scale,
        slepian_nw=args.slepian_nw,
        slepian_k_cutoff=args.slepian_k_cutoff,
        slepian_window=args.slepian_window
    )[0].permute(0, 2, 3, 1).cpu().numpy()

    export_to_video(
        video, os.path.join(args.output_folder, f"output_{prompt_index:03d}.mp4"), fps=16)
