#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer. FFT -> low-mode linear transform -> IFFT.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        # store real/imag parts explicitly (…, 2)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2)
        )

    @staticmethod
    def compl_mul2d(a, b):
        # a: (B, C_in, m1, m2, 2); b: (C_in, C_out, m1, m2, 2)
        ar, ai = a[..., 0], a[..., 1]
        br, bi = b[..., 0], b[..., 1]
        real = torch.einsum("bixy,ioxy->boxy", ar, br) - torch.einsum("bixy,ioxy->boxy", ai, bi)
        imag = torch.einsum("bixy,ioxy->boxy", ar, bi) + torch.einsum("bixy,ioxy->boxy", ai, br)
        return torch.stack([real, imag], dim=-1)

    def forward(self, x):
        """
        x: (B, C_in, H, W)
        """
        B, C, H, W = x.shape

        # rFFT
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)  # (B,C,H,W//2+1,2)

        # allocate output spectrum
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1, 2,
            device=x.device, dtype=x.dtype
        )

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights[:, :, :m1, :m2]
        )

        out_ft_c = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x = torch.fft.irfft2(out_ft_c, s=(H, W), dim=(-2, -1), norm="ortho")
        return x


class FNOBlock2D(nn.Module):
    """
    Minimal 4-layer FNO block with optional padding/cropping to reduce FFT wrap-around artifacts.

    Inputs:
      x: (B, in_channels=2, H, W) -> [IC_normed, Source_normed]

    Output:
      y: (B, out_channels=1, H, W) in normalized space

    Notes:
      - Use masked loss outside this module (loss only computed on mask==1).
      - For data: replace NaNs outside mask with 0 AFTER normalization, before feeding the model.
    """
    def __init__(
        self,
        modes1=32,
        modes2=32,
        width=128,
        in_channels=2,
        out_channels=1,
        padding: int = 16,   # pixels; 0 disables
    ):
        super().__init__()
        self.ACT = F.relu
        self.width = width
        self.padding = int(padding)

        self.fc0 = nn.Conv2d(in_channels, width, kernel_size=1)

        self.s1 = SpectralConv2d(width, width, modes1, modes2)
        self.w1 = nn.Conv2d(width, width, kernel_size=1)

        self.s2 = SpectralConv2d(width, width, modes1, modes2)
        self.w2 = nn.Conv2d(width, width, kernel_size=1)

        self.s3 = SpectralConv2d(width, width, modes1, modes2)
        self.w3 = nn.Conv2d(width, width, kernel_size=1)

        self.s4 = SpectralConv2d(width, width, modes1, modes2)
        self.w4 = nn.Conv2d(width, width, kernel_size=1)

        self.proj1 = nn.Conv2d(width, 256, kernel_size=1)
        self.proj2 = nn.Conv2d(256, out_channels, kernel_size=1)

    def _pad(self, x):
        if self.padding <= 0:
            return x
        p = self.padding
        # pad last two dims: (left,right,top,bottom)
        return F.pad(x, (p, p, p, p), mode="constant", value=0.0)

    def _crop(self, x, H, W):
        if self.padding <= 0:
            return x
        p = self.padding
        return x[..., p:p+H, p:p+W]

    def forward(self, x):
        """
        x: (B, in_channels=2, H, W) -> [IC_normed, Source_normed]
        returns: (B, out_channels=1, H, W)
        """
        B, C, H, W = x.shape

        x = self.fc0(x)
        x = self._pad(x)

        x1 = self.s1(x)
        x  = self.ACT(x1 + self.w1(x))

        x1 = self.s2(x)
        x  = self.ACT(x1 + self.w2(x))

        x1 = self.s3(x)
        x  = self.ACT(x1 + self.w3(x))

        x1 = self.s4(x)
        x  = x1 + self.w4(x)

        x  = self.ACT(self.proj1(x))
        x  = self.proj2(x)

        # crop back to original resolution
        x = self._crop(x, H, W)
        return x


class SingleFNO(nn.Module):
    """
    Single FNO taking [IC_normed, Source_normed] -> prediction (normalized space).
    """
    def __init__(
        self,
        modes1: int = 32,
        modes2: int = 32,
        width: int = 128,
        padding: int = 16,
    ):
        super().__init__()
        self.fno = FNOBlock2D(
            modes1=modes1,
            modes2=modes2,
            width=width,
            in_channels=2,
            out_channels=1,
            padding=padding,
        )

    def forward(self, x):
        """
        x: (B,2,H,W) = [IC_normed, Source_normed]
        returns:
          pred: (B,1,H,W) in normalized space
        """
        return self.fno(x)


# ---------------- Test / smoke-check ----------------
if __name__ == "__main__":
    B, H, W = 2, 256, 256
    x = torch.randn(B, 2, H, W)  # [IC_normed, Source_normed]
    model = SingleFNO(modes1=32, modes2=32, width=128, padding=16)

    with torch.no_grad():
        y = model(x)
    print("Output shape:", tuple(y.shape))  # (B,1,H,W)
