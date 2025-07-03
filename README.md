### This is currently in the pretraining phase... Not ready for production environment

# Diffusion Model Training

This project now uses a diffusion model (DDPM) with Hugging Face diffusers.

## How to train

1. Set your dataset path in `config.py` (variable `DATASET`).

2. Setup virtual environment: 
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run:

```bash
python3 diffusion_train.py
```

Outputs and downscaled images will be saved in the `runs/` folder.

## Requirements
- diffusers
- accelerate
- torch
- torchvision
- PIL

## Notes
- The GAN code has been replaced by a minimal DDPM pipeline.
- You can further customize the U-Net, scheduler, and training loop as needed.
