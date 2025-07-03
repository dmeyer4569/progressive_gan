import torch
from networks.generator import Generator
from networks.discriminator import Discriminator
from config import *

# Set parameters from config
z_dim = Z_DIM
batch_size = BATCH_SIZE

# Instantiate generator and discriminator
gen = Generator(z_dim=z_dim)
disc = Discriminator()

# Create random latent vector
z = torch.randn(batch_size, z_dim)

# Forward pass through generator (steps=0 for 4x4 output)
img = gen(z, steps=0)

# Forward pass through discriminator
output = disc(img)

print("Discriminator output:", output)
print("Generated image shape:", img.shape)