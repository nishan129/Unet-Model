import torch
import torch.nn as nn
from Unet.base_model import DoubleConv, DownSample, UpSample





class ColorConditionalUNet(nn.Module):
    """U-Net with color conditioning"""
    
    def __init__(self, n_channels=3, n_classes=3, n_colors=8, bilinear=True):
        super(ColorConditionalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_colors = n_colors
        self.bilinear = bilinear
        
        # Color embedding
        self.color_embedding = nn.Embedding(n_colors, 64)
        
        # We'll add color information as additional channels
        self.inc = DoubleConv(n_channels + 1, 64)  # +1 for color channel
        
        # Encoder (Contracting Path)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownSample(512, 1024 // factor)
        
        # Decoder (Expansive Path)
        self.up1 = UpSample(1024, 512 // factor, bilinear)
        self.up2 = UpSample(512, 256 // factor, bilinear)
        self.up3 = UpSample(256, 128 // factor, bilinear)
        self.up4 = UpSample(128, 64, bilinear)
        
        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x, color_idx):
        batch_size, _, height, width = x.shape
        
        # Create color channel
        # Color embedding gives us a vector, we need to broadcast it to image dimensions
        color_embed = self.color_embedding(color_idx)  # [batch_size, 64]
        
        # Simple approach: create a single channel with color index value
        color_channel = color_idx.float().view(batch_size, 1, 1, 1)
        color_channel = color_channel.expand(batch_size, 1, height, width) / (self.n_colors - 1)  # Normalize
        
        # Concatenate image with color channel
        x_with_color = torch.cat([x, color_channel], dim=1)
        
        # Encoder
        x1 = self.inc(x_with_color)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits