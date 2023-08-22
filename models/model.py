import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None
        
        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8
        
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.use_pixelnorm = use_pixelnorm
    
    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pixelnorm else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pixelnorm else x
        return x

class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=1):
        super().__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, kernel_size=4, stride=1, padding=0), # 1x1 -> 4x4
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        
        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList([self.initial_rgb])
        
        for i in range(len(factors) - 1):
            # factors[i] -> factors[i+1]
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))
        
    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):
        out = self.initial(x) # 4x4
        
        if steps == 0:
            return self.initial_rgb(out)
        
        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='bicubic', align_corners=False)
            out = self.prog_blocks[step](upscaled)
            
        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)

class Discriminator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels = 1):
        super().__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)
        
        for i in range(len(factors) - 1, 0, -1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i-1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c, use_pixelnorm=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_c, kernel_size=1, stride=1, padding=0))
        
        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1)
        )
    
    def fade_in(self, alpha, downscaled, out):
        return alpha  * out + (1 - alpha) * downscaled
    
    def minibach_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat((x, batch_statistics), dim=1)
    
    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))
        
        if steps == 0:
            out = self.minibach_std(out)
            return self.final_block(out).view(out.shape[0], -1)
        
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)
        
        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)
            
        out = self.minibach_std(out)
        
        return self.final_block(out).view(out.shape[0], -1)
    
if __name__ == "__main__":
    Z_DIM = 50
    IN_CHANNELS = 256
    gen = Generator(Z_DIM, IN_CHANNELS, img_channels=1)
    critic = Discriminator(Z_DIM, IN_CHANNELS, img_channels=1)
    
    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(np.log2(img_size / 4))
        x = torch.randn((1, Z_DIM, 1, 1))
        z = gen(x, alpha=0.5, steps=num_steps)
        assert z.shape == (1, 1, img_size, img_size), f"Bad image size {z.shape}"
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1), f"Bad output size {out.shape}"
        print(f"Success! At img size: {img_size}")




# class GeneratorBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#         super(GeneratorBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
        

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x

# class Generator(pl.LightningModule):
#     def __init__(self, z_dim=128):
#         super(Generator, self).__init__()
#         self.z_dim = z_dim

#         # Initial block to start from 4x4 resolution
#         self.initial_block = nn.Sequential(
#             nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0),
#             nn.BatchNorm2d(512),
#             nn.ReLU()
#         )

#         # Blocks for progressive growing
#         self.blocks = nn.ModuleList([
#             GeneratorBlock(512, 256),
#             GeneratorBlock(256, 128),
#             GeneratorBlock(128, 64),
#             GeneratorBlock(64, 32),
#             GeneratorBlock(32, 16),
#             GeneratorBlock(16, 8),
#             GeneratorBlock(8, 4),
#             GeneratorBlock(4, 1) # Final block to generate heightmap
#         ])

#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

#     def forward(self, z, depth):
#         print(f"Input for generator: {z.size()}")
#         x = self.initial_block(z)
#         for block in self.blocks[:depth]:
#             print(f"Before block: {x.size()}")
#             x = self.upsample(x)
#             x = block(x)
#             print(f"After block: {x.size()}")
#         print(f"Output for generator: {x.size()}") 
#         return x

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, betas=(0.5, 0.999))
#         return optimizer
