"""Imports"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Z_DIM, BASE_CHANNELS, IMAGE_CHANNELS

class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, base_channels=BASE_CHANNELS, image_channels=IMAGE_CHANNELS):
        super(Generator, self).__init__()
        # Initial 4x4 block
        self.initial_conv = nn.ConvTranspose2d(z_dim, base_channels, 4, 1, 0)
        self.initial_bn = nn.BatchNorm2d(base_channels)
        self.initial_act = nn.LeakyReLU(0.2)
        # ToRGB layer for 4x4
        self.to_rgb = nn.Conv2d(base_channels, image_channels, 1)
        # Store progressive blocks
        self.progressive_blocks = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList([self.to_rgb])
        self.current_resolution = 4

    def forward(self, z, alpha=1.0, steps=0):
        x = self.initial_conv(z.view(z.size(0), z.size(1), 1, 1))
        x = self.initial_bn(x)
        x = self.initial_act(x)
        if steps == 0:
            img = self.to_rgb(x)
            return torch.tanh(img)
        # Progressive growing
        for i in range(steps):
            x = self.progressive_blocks[i](x)
        img_new = self.to_rgb_layers[steps](x)
        if alpha < 1.0 and steps > 0:
            x_prev = F.interpolate(x, scale_factor=0.5, mode='nearest')
            img_prev = self.to_rgb_layers[steps-1](x_prev)
            img_new = alpha * img_new + (1 - alpha) * img_prev
        return torch.tanh(img_new)

    def add_block(self, in_channels, out_channels, image_channels=3):
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        self.progressive_blocks.append(block)
        self.to_rgb_layers.append(nn.Conv2d(out_channels, image_channels, 1))
        self.current_resolution *= 2
