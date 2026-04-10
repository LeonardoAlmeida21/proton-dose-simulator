"""
src/model.py
------------
1D and 2D U-Net architectures for proton dose distribution denoising.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
# 1D U-Net Building Blocks
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch,  out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled   = self.pool(features)
        return pooled, features


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet1D(nn.Module):
    def __init__(self, base_ch: int = 32, depth: int = 4):
        super().__init__()
        self.encoders = nn.ModuleList()
        in_ch = 1
        for i in range(depth):
            out_ch = base_ch * (2 ** i)
            self.encoders.append(DownBlock(in_ch, out_ch))
            in_ch = out_ch

        btn_ch = base_ch * (2 ** depth)
        self.bottleneck = ConvBlock(in_ch, btn_ch)

        self.decoders = nn.ModuleList()
        in_ch = btn_ch
        for i in reversed(range(depth)):
            out_ch = base_ch * (2 ** i)
            self.decoders.append(UpBlock(in_ch, out_ch))
            in_ch = out_ch

        self.output_conv = nn.Conv1d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        return F.relu(self.output_conv(x))


# ══════════════════════════════════════════════════════════════════════════════
# 2D U-Net Building Blocks
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock2D(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled   = self.pool(features)
        return pooled, features


class UpBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock2D(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """
    2D U-Net for proton dose distribution denoising.
    Input : tensor [B, 1, H, W]
    Output: tensor [B, 1, H, W]
    """
    def __init__(self, base_ch: int = 32, depth: int = 4):
        super().__init__()
        self.encoders = nn.ModuleList()
        in_ch = 1
        for i in range(depth):
            out_ch = base_ch * (2 ** i)
            self.encoders.append(DownBlock2D(in_ch, out_ch))
            in_ch = out_ch

        btn_ch = base_ch * (2 ** depth)
        self.bottleneck = ConvBlock2D(in_ch, btn_ch)

        self.decoders = nn.ModuleList()
        in_ch = btn_ch
        for i in reversed(range(depth)):
            out_ch = base_ch * (2 ** i)
            self.decoders.append(UpBlock2D(in_ch, out_ch))
            in_ch = out_ch

        self.output_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        return F.relu(self.output_conv(x))


if __name__ == "__main__":
    t1 = sum(p.numel() for p in UNet1D().parameters() if p.requires_grad)
    t2 = sum(p.numel() for p in UNet2D().parameters() if p.requires_grad)
    print(f"UNet1D params: {t1:,}")
    print(f"UNet2D params: {t2:,}")
