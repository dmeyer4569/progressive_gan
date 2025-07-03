import torch
from networks.generator import Generator

# Set parameters
z_dim = 512
batch_size = 4

# Instantiate generator
gen = Generator(z_dim=z_dim)

# Create random latent vector
z = torch.randn(batch_size, z_dim)

# Forward pass (steps=0 for 4x4 output)
img = gen(z, steps=0)
print("Output shape:", img.shape)