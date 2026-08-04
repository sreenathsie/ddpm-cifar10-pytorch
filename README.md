# DDPM CIFAR-10 (PyTorch, from scratch)

A minimal, unconditional Denoising Diffusion Probabilistic Model (Ho et al., 2020) implemented and trained from scratch on CIFAR-10, on a single Colab T4 GPU. No pretrained weights, no diffusers library — the U-Net, noise schedule, training loop, and both samplers (DDPM + DDIM) are all implemented directly.

## Results

Trained for 200 epochs (~390 steps/epoch, ~1.5 min/epoch, ~5 hours total on T4).

| Forward diffusion process | DDPM samples (1000 steps) | DDIM samples (50 steps) |
|---|---|---|
| ![forward](assets/forward_diffusion.png) | ![ddpm](assets/ddpm_samples.png) | ![ddim](assets/ddim_samples.png) |

DDIM recovers comparable sample quality in 20x fewer steps, confirming the sampler reformulation is correct.

## Architecture

- **U-Net**, ~40M parameters
  - Base channels: 64, channel multipliers: (1, 2, 2, 2) → resolutions 32→16→8→4
  - 2 ResNet blocks per resolution level, self-attention at 8×8
  - Sinusoidal timestep embedding injected into every ResBlock
- **Diffusion process**: 1000 timesteps, linear beta schedule (1e-4 → 0.02)
- **Training objective**: simple noise-prediction MSE (`ε`-prediction, as in the original DDPM paper)
- **EMA** of weights (decay 0.9999) used for sampling

## Repo structure

```
.
├── ddpm_cifar10.ipynb       # Full notebook: data, model, training, sampling
├── assets/                  # Sample outputs (forward process, DDPM/DDIM grids)
└── README.md
```

## Training details

- Dataset: CIFAR-10, 32×32, normalized to [-1, 1]
- Batch size: 128, mixed precision (fp16/bf16)
- Optimizer: AdamW, lr 2e-4
- Checkpointed every epoch (Google Drive) to survive Colab session timeouts — training resumes automatically from the latest checkpoint

## Run it yourself

Open `ddpm_cifar10.ipynb` in Google Colab, set runtime to a GPU (T4 or better), and run top to bottom. CIFAR-10 downloads automatically via `torchvision`.

## What this implements (and why)

This project exists to understand the mechanics of diffusion models directly, rather than fine-tuning an existing one:

- **Forward process**: closed-form noising to any timestep `t` via `x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε`
- **Reverse process**: iterative denoising using the model's noise prediction at each step
- **DDIM**: a deterministic, faster alternative sampling procedure using the same trained model — same weights, different sampling math

## Next steps

- **Class-conditioning**: condition the U-Net on CIFAR-10's 10 class labels (embedding added to the timestep embedding pathway) — the stepping stone toward text-conditioned generation
- **Text-conditioning**: swap class embedding for a frozen text encoder (e.g. CLIP) + cross-attention, on a narrow-domain captioned dataset
- Cosine beta schedule (known to improve sample quality over linear)

## References

- Ho, Jain, Abbeel. ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239), 2020.
- Song, Meng, Ermon. ["Denoising Diffusion Implicit Models"](https://arxiv.org/abs/2010.02502), 2020 (DDIM).
