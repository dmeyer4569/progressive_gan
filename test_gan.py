# Test the full data/optim/loss pipeline

import torch
from dataloader import get_dataloader
from optim_and_loss import gen, disc, optimizer_g, optimizer_d, criterion
from config import Z_DIM
import torchvision.utils as vutils
import torchvision.transforms as T

import os
import shutil
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Create run directory structure
def create_run_dirs():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("runs", now)
    outputs_dir = os.path.join(base_dir, "outputs")
    epochs_dir = os.path.join(base_dir, "epochs")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(epochs_dir, exist_ok=True)
    # Save config.py
    shutil.copy("config.py", os.path.join(base_dir, "config.py"))
    # Create log file
    log_path = os.path.join(base_dir, "log.txt")
    with open(log_path, "w") as f:
        f.write(f"Run started at {now}\n")
    return base_dir, outputs_dir, epochs_dir, log_path

def test_pipeline():
    base_dir, outputs_dir, epochs_dir, log_path = create_run_dirs()
    dataloader = get_dataloader()
    batch = next(iter(dataloader))
    batch = batch.to(device)
    print("Loaded batch shape:", batch.shape)

    # Forward pass through generator
    z = torch.randn(batch.shape[0], Z_DIM, device=device)
    fake_imgs = gen(z)
    print("Generated fake images shape:", fake_imgs.shape)

    # Forward pass through discriminator
    real_out = disc(batch)
    fake_out = disc(fake_imgs)
    print("Discriminator output (real):", real_out)
    print("Discriminator output (fake):", fake_out)

    # Save a grid of generated images, resized to 256x256
    grid = vutils.make_grid(fake_imgs, nrow=4, normalize=True, scale_each=True)
    to_pil = T.ToPILImage()
    grid_img = to_pil(grid)
    grid_img = grid_img.resize((256, 256), resample=2)
    out_path = os.path.join(outputs_dir, "generated_grid.png")
    grid_img.save(out_path)
    print(f"Saved generated images grid as {out_path}")
    # Log output
    with open(log_path, "a") as f:
        f.write(f"Saved generated images grid as {out_path}\n")

if __name__ == "__main__":
    test_pipeline()