
# Test the full data/optim/loss pipeline

import torch
from dataloader import get_dataloader
from optim_and_loss import gen, disc, optimizer_g, optimizer_d, criterion, device
from config import Z_DIM

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

if __name__ == "__main__":
    test_pipeline()