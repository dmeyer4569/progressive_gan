import torch
import torch.nn as nn
from networks.generator import Generator
from networks.discriminator import Discriminator
from config import LEARNING_RATE, Z_DIM

# Initialize models
gen = Generator(z_dim=Z_DIM)
print("Generator initialized")
disc = Discriminator()
print("Discriminator initialized.")

# Optimizers
optimizer_g = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
optimizer_d = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Optionally, move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen = gen.to(device)
disc = disc.to(device)
