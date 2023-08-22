import torch
import torch.nn as nn
import pytorch_lightning as pl

class CriticBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CriticBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x

class Critic(pl.LightningModule):
    def __init__(self):
        super(Critic, self).__init__()

        # Blocks for progressive downsampling
        self.blocks = nn.ModuleList([
            CriticBlock(1, 4),
            CriticBlock(4, 8),
            CriticBlock(8, 16),
            CriticBlock(16, 32),
            CriticBlock(32, 64),
            CriticBlock(64, 128),
            CriticBlock(128, 256),
            CriticBlock(256, 512)
        ])

        self.downsample = nn.AvgPool2d(2)

        # Final layer to produce a single value
        self.final_layer = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Flatten()
        )

    def forward(self, x, depth):
        print (f"Size of x at start: {x.size()}")
        for block in reversed(self.blocks[:depth]):
            x = block(x)
            x = self.downsample(x)
            print (f"Size of x: {x.size()}")
        x = self.final_layer(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return optimizer
