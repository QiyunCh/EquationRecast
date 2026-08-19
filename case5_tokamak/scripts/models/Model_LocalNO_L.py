#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from neuralop.models import LocalNO



# ============================================================
# CONFIG
# ============================================================

@dataclass
class LocalNOConfig:
    in_channels: int = 2
    out_channels: int = 1
    image_size: tuple[int, int] = (256, 256)

    n_modes: tuple[int, int] = (48, 48)
    hidden_channels: int = 128
    n_layers: int = 4

    diff_layers: bool = True
    disco_layers: bool = True
    disco_kernel_shape: tuple[int, int] = (2, 4)
    fin_diff_kernel_size: int = 3

    use_channel_mlp: bool = False
    norm: str = "instance_norm"
    domain_padding: float = 0.05
    positional_embedding: str = "grid"
    local_no_skip: str = "linear"

    conv_padding_mode: str = "zeros"
    projection_channel_ratio: float = 2.0
    lifting_channel_ratio: float = 2.0


# ============================================================
# MODEL
# ============================================================

class SingleLocalNO(nn.Module):
    def __init__(self, cfg: LocalNOConfig | None = None):
        super().__init__()
        self.cfg = cfg or LocalNOConfig()

        self.model = LocalNO(
            n_modes=self.cfg.n_modes,
            in_channels=self.cfg.in_channels,
            out_channels=self.cfg.out_channels,
            hidden_channels=self.cfg.hidden_channels,
            default_in_shape=self.cfg.image_size,
            n_layers=self.cfg.n_layers,
            diff_layers=self.cfg.diff_layers,
            disco_layers=self.cfg.disco_layers,
            disco_kernel_shape=list(self.cfg.disco_kernel_shape),
            fin_diff_kernel_size=self.cfg.fin_diff_kernel_size,
            use_channel_mlp=self.cfg.use_channel_mlp,
            norm=self.cfg.norm,
            domain_padding=self.cfg.domain_padding,
            positional_embedding=self.cfg.positional_embedding,
            local_no_skip=self.cfg.local_no_skip,
            conv_padding_mode=self.cfg.conv_padding_mode,
            projection_channel_ratio=self.cfg.projection_channel_ratio,
            lifting_channel_ratio=self.cfg.lifting_channel_ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================
# UTIL
# ============================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model():
    return SingleLocalNO()


# ============================================================
# MAIN: SMOKE TEST
# ============================================================

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model().to(device)
    model.train()

    B, H, W = 2, 256, 256

    x = torch.randn(B, 2, H, W, device=device)
    y_true = torch.randn(B, 1, H, W, device=device)

    mask = (torch.rand(B, 1, H, W, device=device) > 0.2).float()

    y_pred = model(x)

    loss = ((y_pred - y_true) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
    loss.backward()

    print("=" * 60)
    print("LocalNO smoke test passed")
    print("=" * 60)
    print(f"device   : {device}")
    print(f"input    : {tuple(x.shape)}")
    print(f"output   : {tuple(y_pred.shape)}")
    print(f"loss     : {loss.item():.6e}")
    print(f"params   : {count_parameters(model):,}")