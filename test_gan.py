# Test the full data/optim/loss pipeline

import torch
from dataloader import get_dataloader
from optim_and_loss import gen, disc, optimizer_g, optimizer_d, criterion
from config import Z_DIM
import torchvision.utils as vutils
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def test_pipeline():
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
    grid_img.save("generated_grid.png")
    print("Saved generated images grid as generated_grid.png")

if __name__ == "__main__":
    test_pipeline()