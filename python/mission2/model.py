"""Sinusoidal timestep embedding and a small time-conditioned noise U-Net."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if not isinstance(dim, int) or dim < 4 or dim % 2 != 0:
            raise ValueError("dim must be an even integer >= 4")
        self.dim = dim
        half_dim = dim // 2
        # Log-spaced frequencies: 1 down to 1/10000, inclusive.
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, dtype=torch.float32)
            / (half_dim - 1)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Convert integer timestep indices [B] into float features [B, dim]."""
        if t.ndim != 1 or t.dtype != torch.long:
            raise ValueError("Expected t with shape [B] and dtype torch.long")
        if t.device != self.frequencies.device:
            raise ValueError("t and embedding buffers must be on the same device")
        # [B,1] * [dim/2] -> [B,dim/2] through broadcasting.
        phases = t.to(dtype=self.frequencies.dtype).unsqueeze(1) * self.frequencies
        # Each timestep gets sin features followed by cos features.
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)


class ResidualBlock(nn.Module):
    """Transform image features and add a projected timestep vector."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # A 1x1 convolution aligns channels before residual addition if needed.
        self.shortcut = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # [B,C] -> [B,C,1,1], broadcast across the spatial dimensions.
        h = h + self.time_projection(time_features)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class UNet(nn.Module):
    """Predict epsilon [B,1,28,28] from a noisy image and timestep [B]."""

    def __init__(self, base_channels: int = 32, time_embedding_dim: int = 32):
        super().__init__()
        if base_channels < 8 or base_channels % 8 != 0:
            raise ValueError("base_channels must be a positive multiple of 8")
        c = base_channels
        time_dim = c * 4
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_conv = nn.Conv2d(1, c, kernel_size=3, padding=1)
        self.encoder1 = ResidualBlock(c, c, time_dim)
        # Strided 3x3 convolutions yield 28 -> 14 -> 7.
        self.down1 = nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1)
        self.encoder2 = ResidualBlock(c, c * 2, time_dim)
        self.down2 = nn.Conv2d(c * 2, c * 2, kernel_size=3, stride=2, padding=1)
        self.encoder3 = ResidualBlock(c * 2, c * 4, time_dim)
        self.middle = ResidualBlock(c * 4, c * 4, time_dim)
        self.up2_conv = nn.Conv2d(c * 4, c * 2, kernel_size=3, padding=1)
        self.decoder2 = ResidualBlock(c * 4, c * 2, time_dim)
        self.up1_conv = nn.Conv2d(c * 2, c, kernel_size=3, padding=1)
        self.decoder1 = ResidualBlock(c * 2, c, time_dim)
        self.output_norm = nn.GroupNorm(8, c)
        self.output_conv = nn.Conv2d(c, 1, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 4 or x_t.shape[1:] != (1, 28, 28):
            raise ValueError("Expected x_t with shape [B,1,28,28]")
        if t.shape != (x_t.shape[0],) or t.dtype != torch.long:
            raise ValueError("Expected t with shape [B] and dtype torch.long")
        if x_t.device != t.device or x_t.device != self.input_conv.weight.device:
            raise ValueError("x_t, t and U-Net must be on the same device")

        time_features = self.time_embedding(t)
        skip1 = self.encoder1(self.input_conv(x_t), time_features)
        skip2 = self.encoder2(self.down1(skip1), time_features)
        h = self.encoder3(self.down2(skip2), time_features)
        h = self.middle(h, time_features)

        # Match saved spatial sizes explicitly; nearest-neighbor resize adds no parameters.
        h = self.up2_conv(F.interpolate(h, size=skip2.shape[-2:], mode="nearest"))
        h = self.decoder2(torch.cat([h, skip2], dim=1), time_features)
        h = self.up1_conv(F.interpolate(h, size=skip1.shape[-2:], mode="nearest"))
        h = self.decoder1(torch.cat([h, skip1], dim=1), time_features)
        # Noise is unbounded, so the output has no sigmoid/tanh activation.
        return self.output_conv(F.silu(self.output_norm(h)))
