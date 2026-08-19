#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data.py - Simplified data loading module for FNO training.

Simplifications vs original:
  1. masked_mse assumes fixed shapes, removes complex shape checking
  2. Shared mask cached in RAM, not re-read from HDF5 each sample
  3. Removed DDP code (can be added separately if needed)
  4. Streamlined code (~220 lines vs ~650 lines)

Returns (x, y, mask) for compatibility with existing test scripts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple, Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CONFIG
# =============================================================================

DATA_H5_PATH = Path("Data_ML_Merged_No_t000_t001.h5")
STATS_JSON_PATH = Path("stats_train_Median.json")

TRAIN_FRAC = 0.80
SHUFFLE_SEED = 816
BATCH_SIZE = 8
NUM_WORKERS = 0

EPS = 1e-12


# =============================================================================
# STATS LOADING
# =============================================================================

def load_stats(path: Path) -> Dict[str, Any]:
    """Load and validate normalization statistics."""
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    
    # Simple validation
    assert "scales" in stats and "mu_sigma" in stats
    assert all(k in stats["scales"] for k in ["T0_fixed", "q0", "y0"])
    assert all(k in stats["mu_sigma"] for k in ["X_T", "X_q", "Y"])
    
    return stats


# =============================================================================
# NORMALIZATION (NumPy)
# =============================================================================

def normalize_ic(T: np.ndarray, stats: Dict) -> np.ndarray:
    """Normalize IC: X_T = (log1p(T / T0) - mu) / sigma"""
    T = np.maximum(np.asarray(T, dtype=np.float64), 0.0)  # clamp negatives
    T0 = stats["scales"]["T0_fixed"]
    mu = stats["mu_sigma"]["X_T"]["mu"]
    sigma = stats["mu_sigma"]["X_T"]["sigma"]
    return ((np.log1p(T / T0) - mu) / sigma).astype(np.float32)


def normalize_source(q: np.ndarray, stats: Dict) -> np.ndarray:
    """Normalize Source: X_q = (asinh(q / q0) - mu) / sigma"""
    q = np.asarray(q, dtype=np.float64)
    q0 = stats["scales"]["q0"]
    mu = stats["mu_sigma"]["X_q"]["mu"]
    sigma = stats["mu_sigma"]["X_q"]["sigma"]
    return ((np.arcsinh(q / q0) - mu) / sigma).astype(np.float32)


def normalize_label(dT: np.ndarray, stats: Dict) -> np.ndarray:
    """Normalize Label: Y = (asinh(dT / y0) - mu) / sigma"""
    dT = np.asarray(dT, dtype=np.float64)
    y0 = stats["scales"]["y0"]
    mu = stats["mu_sigma"]["Y"]["mu"]
    sigma = stats["mu_sigma"]["Y"]["sigma"]
    return ((np.arcsinh(dT / y0) - mu) / sigma).astype(np.float32)


# =============================================================================
# DENORMALIZATION (Torch)
# =============================================================================

def denorm_deltaT(Y_norm: torch.Tensor, stats: Dict) -> torch.Tensor:
    """Denormalize prediction: dT = y0 * sinh(Y_norm * sigma + mu)"""
    y0 = stats["scales"]["y0"]
    mu = stats["mu_sigma"]["Y"]["mu"]
    sigma = stats["mu_sigma"]["Y"]["sigma"]
    return y0 * torch.sinh(Y_norm * sigma + mu)


def denorm_ic(X_norm: torch.Tensor, stats: Dict) -> torch.Tensor:
    """Denormalize IC: T = T0 * expm1(X_norm * sigma + mu)"""
    T0 = stats["scales"]["T0_fixed"]
    mu = stats["mu_sigma"]["X_T"]["mu"]
    sigma = stats["mu_sigma"]["X_T"]["sigma"]
    return T0 * torch.expm1(X_norm * sigma + mu)


# =============================================================================
# LOSS
# =============================================================================

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Simplified Masked MSE.
    
    Args:
        pred:   (B, 1, H, W)
        target: (B, 1, H, W)
        mask:   (B, 1, H, W) or broadcastable shape
    
    Returns:
        Scalar loss
    """
    diff_sq = (pred - target) ** 2
    return (mask * diff_sq).sum() / (mask.sum() + EPS)


# =============================================================================
# TRAIN/VAL SPLIT
# =============================================================================

def compute_train_val_split(n_total: int, train_frac: float = TRAIN_FRAC, seed: int = SHUFFLE_SEED):
    """Compute train/val indices (matches Compute_train_stats.py convention)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    
    n_train = int(math.floor(train_frac * n_total))
    return np.sort(idx[:n_train]), np.sort(idx[n_train:])


# =============================================================================
# DATASET
# =============================================================================

class FNODataset(Dataset):
    """
    Simplified Dataset with cached shared mask.
    
    Returns:
        x: (2, H, W)  [IC_norm, Source_norm]
        y: (1, H, W)  [Label_norm]
        m: (1, H, W)  [mask, cached in RAM]
    """
    
    def __init__(self, h5_path: Path, stats: Dict, indices: np.ndarray):
        self.h5_path = Path(h5_path)
        self.stats = stats
        self.indices = indices.astype(np.int64)
        self._h5 = None
        
        # Load and cache mask once
        with h5py.File(self.h5_path, "r") as f:
            mask_raw = np.asarray(f["Mask"], dtype=np.float32)
            # Handle (H,W) shared or (N,H,W) per-sample format
            if mask_raw.ndim == 2:
                self._mask = mask_raw  # (H, W)
            else:
                self._mask = mask_raw[0]  # Take first, assume all same
    
    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5
    
    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        f = self._open()
        
        # Load raw data
        ic = np.asarray(f["IC"][idx], dtype=np.float32)
        src = np.asarray(f["Source"][idx], dtype=np.float32)
        label = np.asarray(f["Label"][idx], dtype=np.float32)
        
        # NaN -> 0 (mask ensures they don't affect loss)
        ic = np.nan_to_num(ic, nan=0.0, posinf=0.0, neginf=0.0)
        src = np.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0)
        label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize
        ic_norm = normalize_ic(ic, self.stats)
        src_norm = normalize_source(src, self.stats)
        label_norm = normalize_label(label, self.stats)
        
        # Stack and reshape
        x = np.stack([ic_norm, src_norm], axis=0)  # (2, H, W)
        y = label_norm[np.newaxis, ...]             # (1, H, W)
        m = self._mask[np.newaxis, ...]             # (1, H, W) from cache
        
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(m.copy()),  # copy to avoid issues with shared memory
        )


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def make_dataloaders(
    data_path: Path = DATA_H5_PATH,
    stats_path: Path = STATS_JSON_PATH,
    batch_size: int = BATCH_SIZE,
    val_frac: float = 1.0 - TRAIN_FRAC,
    seed: int = SHUFFLE_SEED,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create training and validation DataLoaders.
    
    Returns:
        train_loader, val_loader, stats
    """
    stats = load_stats(stats_path)
    train_frac = 1.0 - val_frac
    
    # Get dataset size
    with h5py.File(data_path, "r") as f:
        n_total = f["IC"].shape[0]
        H, W = f["IC"].shape[1], f["IC"].shape[2]
    
    # Split indices
    train_idx, val_idx = compute_train_val_split(n_total, train_frac, seed)
    
    print(f"[Data] {data_path.name}: {n_total} samples, {H}x{W}")
    print(f"[Data] Split: {len(train_idx)} train, {len(val_idx)} val (seed={seed})")
    print(f"[Data] T0={stats['scales']['T0_fixed']:.2e}, q0={stats['scales']['q0']:.2e}, y0={stats['scales']['y0']:.2e}")
    
    # Create Datasets
    train_ds = FNODataset(data_path, stats, train_idx)
    val_ds = FNODataset(data_path, stats, val_idx)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    return train_loader, val_loader, stats


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Data.py...")
    
    # Test masked_mse
    B, H, W = 4, 256, 256
    pred = torch.randn(B, 1, H, W)
    target = torch.randn(B, 1, H, W)
    mask = torch.ones(B, 1, H, W)
    
    loss = masked_mse(pred, target, mask)
    print(f"masked_mse test: loss = {loss.item():.4f}")
    
    # Test normalization roundtrip
    stats_mock = {
        "scales": {"T0_fixed": 0.01, "q0": 1.0, "y0": 1.0},
        "mu_sigma": {
            "X_T": {"mu": 0.0, "sigma": 1.0},
            "X_q": {"mu": 0.0, "sigma": 1.0},
            "Y": {"mu": 0.0, "sigma": 1.0},
        }
    }
    
    # IC roundtrip
    ic_raw = np.random.rand(64, 64).astype(np.float32) * 0.1
    ic_norm = normalize_ic(ic_raw, stats_mock)
    ic_back = denorm_ic(torch.from_numpy(ic_norm), stats_mock).numpy()
    ic_norm2 = normalize_ic(ic_back, stats_mock)
    print(f"IC roundtrip max error: {np.max(np.abs(ic_norm2 - ic_norm)):.2e}")
    
    # Label roundtrip
    y_raw = np.random.randn(64, 64).astype(np.float32) * 0.01
    y_norm = normalize_label(y_raw, stats_mock)
    y_back = denorm_deltaT(torch.from_numpy(y_norm), stats_mock).numpy()
    y_norm2 = normalize_label(y_back, stats_mock)
    print(f"Label roundtrip max error: {np.max(np.abs(y_norm2 - y_norm)):.2e}")
    
    print("All tests passed!")