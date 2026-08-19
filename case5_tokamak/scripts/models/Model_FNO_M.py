#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from neuralop.models import FNO


# ============================================================
# CONFIG
# ============================================================

@dataclass
class FNOConfig:
    in_channels: int = 2
    out_channels: int = 1
    image_size: tuple[int, int] = (256, 256)

    # Dense FNO baseline for 256x256 2D fields
    n_modes: tuple[int, int] = (32, 32)
    hidden_channels: int = 64
    n_layers: int = 4

    # Architecture options
    use_channel_mlp: bool = False
    norm: str = "instance_norm"
    domain_padding: float = 0.05
    positional_embedding: str = "grid"
    fno_skip: str = "linear"
    channel_mlp_skip: str = "soft-gating"

    # Width scaling in lifting / projection MLPs
    projection_channel_ratio: float = 2.0
    lifting_channel_ratio: float = 2.0


# ============================================================
# MODEL
# ============================================================

class SingleFNO(nn.Module):
    def __init__(self, cfg: FNOConfig | None = None):
        super().__init__()
        self.cfg = cfg or FNOConfig()

        self.model = FNO(
            n_modes=self.cfg.n_modes,
            in_channels=self.cfg.in_channels,
            out_channels=self.cfg.out_channels,
            hidden_channels=self.cfg.hidden_channels,
            n_layers=self.cfg.n_layers,
            lifting_channel_ratio=self.cfg.lifting_channel_ratio,
            projection_channel_ratio=self.cfg.projection_channel_ratio,
            positional_embedding=self.cfg.positional_embedding,
            norm=self.cfg.norm,
            use_channel_mlp=self.cfg.use_channel_mlp,
            fno_skip=self.cfg.fno_skip,
            channel_mlp_skip=self.cfg.channel_mlp_skip,
            domain_padding=self.cfg.domain_padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================
# UTIL
# ============================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model() -> SingleFNO:
    return SingleFNO()


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
    print("FNO smoke test passed")
    print("=" * 60)
    print(f"device   : {device}")
    print(f"input    : {tuple(x.shape)}")
    print(f"output   : {tuple(y_pred.shape)}")
    print(f"loss     : {loss.item():.6e}")
    print(f"params   : {count_parameters(model):,}")
