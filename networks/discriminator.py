"""Imports"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BASE_CHANNELS, IMAGE_CHANNELS

class Discriminator(nn.Module):
    def __init__(self, base_channels=BASE_CHANNELS, image_channels=IMAGE_CHANNELS):
        super(Discriminator, self).__init__()
        # FromRGB layer for 4x4
        self.from_rgb = nn.Conv2d(image_channels, base_channels, 1)
        self.initial_bn = nn.BatchNorm2d(base_channels)
        self.initial_act = nn.LeakyReLU(0.2)
        # Store progressive blocks
        self.progressive_blocks = nn.ModuleList()
        self.from_rgb_layers = nn.ModuleList([self.from_rgb])
        self.current_resolution = 4
        # Final block to output real/fake score
        self.final_conv = nn.Conv2d(base_channels, 1, 4)

    def forward(self, x, alpha=1.0, steps=0):
        if steps == 0:
            x = self.from_rgb(x)
            x = self.initial_bn(x)
            x = self.initial_act(x)
        else:
            # Blend input from current and previous resolution
            x_new = self.from_rgb_layers[steps](x)
            x_new = self.initial_bn(x_new)
            x_new = self.initial_act(x_new)
            if alpha < 1.0:
                x_down = F.interpolate(x, scale_factor=0.5, mode='nearest')
                x_prev = self.from_rgb_layers[steps-1](x_down)
                x_prev = self.initial_bn(x_prev)
                x_prev = self.initial_act(x_prev)
                x_new = alpha * x_new + (1 - alpha) * x_prev
            x = x_new
            for i in range(steps-1, -1, -1):
                x = self.progressive_blocks[i](x)
        # Final 4x4 block
        x = self.final_conv(x)
        return x.view(x.size(0), 1)

    def add_block(self, in_channels, out_channels, image_channels=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2)
        )
        self.progressive_blocks.insert(0, block)
        self.from_rgb_layers.append(nn.Conv2d(image_channels, in_channels, 1))
        self.current_resolution *= 2
