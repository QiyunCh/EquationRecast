#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute_train_stats.py

Compute normalization statistics for FNO training using Data_ML.h5.

REVISED VERSION:
  - Uses the ENTIRE dataset to calculate normalization factors (q0, y0, mu, sigma),
    not just the training fraction. This ensures consistent normalization regardless
    of train/val split changes.
  - T0 is a FIXED constant: T0_fixed = 100.0 (do not estimate from data).
  - Masked statistics: use only Mask==1 and finite values (NaNs ignored).
  - Supports both:
      * shared 2D mask: Mask shape (H, W)
      * per-sample 3D mask: Mask shape (N, H, W)

Normalization scheme (must match Data.py):
  - X_T = log1p(IC / T0_fixed)
  - X_q = asinh(Source / q0)
  - Y   = asinh(Label / y0)
  - Standardization: (var - mu) / sigma computed on ENTIRE dataset (masked)

Outputs:
  - stats_train.json with keys:
      scales:   { "T0_fixed", "q0", "y0" }
      mu_sigma: { "X_T": {mu, sigma}, "X_q": {mu, sigma}, "Y": {mu, sigma} }
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple

import h5py
import numpy as np


# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH = Path("Data_ML_Merged_No_t000_t001.h5")
OUT_JSON = Path("stats_train.json")

# Fixed physical scale for temperature normalization
T0_FIXED = 0.01

# Scale method for q0 and y0: "median" or "p75"
SCALE_METHOD = "median"

# HDF5 read chunk: samples per block
CHUNK_SAMPLES = 16

# Diagnostics: keep up to this many masked values per channel for percentile checks
DIAG_MAX = 2_000_000


# =============================================================================
# HELPERS
# =============================================================================

EPS = 1e-12


def _masked_values(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract values where mask==1 and value is finite.

    Args:
        arr:  ndarray, any shape
        mask: ndarray, same shape as arr, values 0/1

    Returns:
        1D float array of masked finite values
    """
    m = (mask.astype(bool)) & np.isfinite(arr)
    if not np.any(m):
        return np.array([], dtype=np.float64)
    return arr[m].astype(np.float64, copy=False).ravel()


def _safe_scale_from_abs(vals_abs: np.ndarray, method: str) -> float:
    """
    Compute robust scale from absolute values using median or p75.

    Args:
        vals_abs: 1D array of abs(values)
        method: "median" or "p75"

    Returns:
        Positive float scale
    """
    if vals_abs.size == 0:
        return 1.0

    if method == "median":
        s = float(np.median(vals_abs))
    elif method == "p75":
        s = float(np.percentile(vals_abs, 75))
    else:
        raise ValueError(f"Unknown scale method: {method}")

    if (not np.isfinite(s)) or s <= EPS:
        s = 1.0
    return s


def _init_acc() -> Dict[str, Any]:
    """Initialize accumulator for online mean/variance computation."""
    return {"count": 0, "sum": 0.0, "sumsq": 0.0}


def _update_acc(acc: Dict[str, Any], vals: np.ndarray) -> None:
    """Update accumulator with new values."""
    if vals.size == 0:
        return
    acc["count"] += int(vals.size)
    acc["sum"] += float(np.sum(vals))
    acc["sumsq"] += float(np.sum(vals * vals))


def _finalize_acc(acc: Dict[str, Any]) -> Dict[str, Any]:
    """Compute mean and std from accumulator."""
    if acc["count"] == 0:
        return {"count": 0, "mean": None, "std": None}
    mean = acc["sum"] / acc["count"]
    var = acc["sumsq"] / acc["count"] - mean * mean
    var = max(var, 0.0)
    return {"count": int(acc["count"]), "mean": float(mean), "std": float(math.sqrt(var))}


def _percentile_summary(vals: np.ndarray, pct_list, name: str) -> Dict[str, Any]:
    """Compute percentile summary (for debug / sanity checking)."""
    if vals.size == 0:
        return {"name": name, "count": 0}
    qs = np.percentile(vals, pct_list).tolist()
    return {
        "name": name,
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "percentiles": {str(p): float(q) for p, q in zip(pct_list, qs)},
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _append_diag(diag_list, vals: np.ndarray, limit: int) -> None:
    """Append values to a diagnostic list up to an overall size limit."""
    if vals.size == 0:
        return
    cur = sum(x.size for x in diag_list) if diag_list else 0
    if cur >= limit:
        return
    take = min(vals.size, limit - cur)
    diag_list.append(vals[:take].astype(np.float32, copy=False))


def _get_mask_block(M: h5py.Dataset, block_idx: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Load a mask block and return a uint8 array shaped like the data block.

    Supports:
      - shared 2D mask: (H, W)
      - per-sample 3D mask: (N, H, W)

    Args:
        M: HDF5 dataset "Mask"
        block_idx: array of indices (len = B)
        target_shape: expected (B, H, W)

    Returns:
        mask block as uint8 array with shape target_shape
    """
    if M.ndim == 2:
        mask_2d = np.asarray(M[...], dtype=np.uint8)  # (H,W)
        return np.broadcast_to(mask_2d, target_shape).astype(np.uint8, copy=False)
    if M.ndim == 3:
        return np.asarray(M[block_idx, ...], dtype=np.uint8)  # (B,H,W)
    raise ValueError(f"Unsupported Mask ndim={M.ndim}, shape={M.shape}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH.resolve()}")

    # Basic checks
    with h5py.File(DATA_PATH, "r") as f:
        for k in ["IC", "Source", "Label", "Mask"]:
            if k not in f:
                raise KeyError(f"Missing dataset '{k}' in {DATA_PATH}")
        N = int(f["IC"].shape[0])
        H = int(f["IC"].shape[1])
        W = int(f["IC"].shape[2])

    # Use ALL indices (entire dataset)
    all_idx = np.arange(N)

    print(f"[INFO] Dataset: {DATA_PATH.resolve()}")
    print(f"[INFO] N={N}, HxW={H}x{W}")
    print(f"[INFO] Computing statistics on ENTIRE dataset (N={N} samples)")
    print(f"[INFO] T0_FIXED={T0_FIXED} (fixed), SCALE_METHOD={SCALE_METHOD}")

    # -------------------------------------------------------------------------
    # PASS 1: compute q0 and y0 from ENTIRE dataset (masked)
    # -------------------------------------------------------------------------
    q_abs_parts = []
    y_abs_parts = []

    with h5py.File(DATA_PATH, "r") as f:
        Q = f["Source"]
        Y = f["Label"]
        M = f["Mask"]

        for s in range(0, N, CHUNK_SAMPLES):
            block = all_idx[s:s + CHUNK_SAMPLES]
            q_blk = np.asarray(Q[block, ...], dtype=np.float64)  # (B,H,W)
            y_blk = np.asarray(Y[block, ...], dtype=np.float64)  # (B,H,W)

            m_blk = _get_mask_block(M, block, q_blk.shape)

            qv = _masked_values(q_blk, m_blk)
            yv = _masked_values(y_blk, m_blk)

            if qv.size:
                q_abs_parts.append(np.abs(qv))
            if yv.size:
                y_abs_parts.append(np.abs(yv))

    q_abs = np.concatenate(q_abs_parts) if q_abs_parts else np.array([], dtype=np.float64)
    y_abs = np.concatenate(y_abs_parts) if y_abs_parts else np.array([], dtype=np.float64)

    q0 = _safe_scale_from_abs(q_abs, SCALE_METHOD)
    y0 = _safe_scale_from_abs(y_abs, SCALE_METHOD)

    print(f"[INFO] q0 ({SCALE_METHOD}) = {q0:.6e}")
    print(f"[INFO] y0 ({SCALE_METHOD}) = {y0:.6e}")

    # -------------------------------------------------------------------------
    # PASS 2: compute mu/sigma for transformed variables on ENTIRE dataset (masked)
    # -------------------------------------------------------------------------
    acc_XT = _init_acc()
    acc_Xq = _init_acc()
    acc_Y = _init_acc()

    diag_XT, diag_Xq, diag_Y = [], [], []
    diag_Q_raw, diag_Y_raw = [], []

    with h5py.File(DATA_PATH, "r") as f:
        IC = f["IC"]
        Q = f["Source"]
        Y = f["Label"]
        M = f["Mask"]

        for s in range(0, N, CHUNK_SAMPLES):
            block = all_idx[s:s + CHUNK_SAMPLES]

            ic_blk = np.asarray(IC[block, ...], dtype=np.float64)
            q_blk = np.asarray(Q[block, ...], dtype=np.float64)
            y_blk = np.asarray(Y[block, ...], dtype=np.float64)

            m_blk = _get_mask_block(M, block, ic_blk.shape)

            icv = _masked_values(ic_blk, m_blk)
            qv = _masked_values(q_blk, m_blk)
            yv = _masked_values(y_blk, m_blk)

            # Raw diagnostics
            _append_diag(diag_Q_raw, qv, DIAG_MAX)
            _append_diag(diag_Y_raw, yv, DIAG_MAX)

            # X_T = log1p(IC / T0_FIXED); IC should be >= 0 for log1p
            if icv.size:
                icv = np.maximum(icv, 0.0)
                XT = np.log1p(icv / float(T0_FIXED))
                _update_acc(acc_XT, XT)
                _append_diag(diag_XT, XT, DIAG_MAX)

            # X_q = asinh(Source / q0)
            if qv.size:
                Xq = np.arcsinh(qv / float(q0))
                _update_acc(acc_Xq, Xq)
                _append_diag(diag_Xq, Xq, DIAG_MAX)

            # Y = asinh(Label / y0)
            if yv.size:
                Yt = np.arcsinh(yv / float(y0))
                _update_acc(acc_Y, Yt)
                _append_diag(diag_Y, Yt, DIAG_MAX)

    mu_sigma_XT = _finalize_acc(acc_XT)
    mu_sigma_Xq = _finalize_acc(acc_Xq)
    mu_sigma_Y = _finalize_acc(acc_Y)

    # Sanity: avoid zero std
    for ms in (mu_sigma_XT, mu_sigma_Xq, mu_sigma_Y):
        if ms["std"] is None or (not np.isfinite(ms["std"])) or ms["std"] <= 0:
            ms["std"] = 1.0

    # Diagnostics percentiles
    pct_diag = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    pct_raw = [0, 50, 75, 90, 95, 99, 100]

    XT_all = np.concatenate(diag_XT).astype(np.float64, copy=False) if diag_XT else np.array([], np.float64)
    Xq_all = np.concatenate(diag_Xq).astype(np.float64, copy=False) if diag_Xq else np.array([], np.float64)
    Y_all = np.concatenate(diag_Y).astype(np.float64, copy=False) if diag_Y else np.array([], np.float64)

    Q_raw_all = np.concatenate(diag_Q_raw).astype(np.float64, copy=False) if diag_Q_raw else np.array([], np.float64)
    Y_raw_all = np.concatenate(diag_Y_raw).astype(np.float64, copy=False) if diag_Y_raw else np.array([], np.float64)

    diag_stats = {
        "X_T_transformed": _percentile_summary(XT_all, pct_diag, "X_T = log1p(IC/T0_fixed)"),
        "X_q_transformed": _percentile_summary(Xq_all, pct_diag, "X_q = asinh(Source/q0)"),
        "Y_transformed": _percentile_summary(Y_all, pct_diag, "Y = asinh(Label/y0)"),
        "Source_raw": _percentile_summary(Q_raw_all, pct_raw, "Source (all data, masked)"),
        "Label_raw": _percentile_summary(Y_raw_all, pct_raw, "Label (all data, masked)"),
    }

    out = {
        "dataset": str(DATA_PATH.resolve()),
        "shape": [N, H, W],
        "N_total": int(N),
        "N_used_for_stats": int(N),  # Now using entire dataset
        "stats_computed_on": "entire_dataset",

        "masking": "All stats use only Mask==1 and finite values (NaNs ignored).",

        "scales": {
            "T0_fixed": float(T0_FIXED),
            "q0": float(q0),
            "y0": float(y0),
        },
        "transforms": {
            "X_T": "log1p(IC / T0_fixed)",
            "X_q": "asinh(Source / q0)",
            "Y": "asinh(Label / y0)",
        },
        "mu_sigma": {
            "X_T": {"mu": mu_sigma_XT["mean"], "sigma": mu_sigma_XT["std"], "count": mu_sigma_XT["count"]},
            "X_q": {"mu": mu_sigma_Xq["mean"], "sigma": mu_sigma_Xq["std"], "count": mu_sigma_Xq["count"]},
            "Y": {"mu": mu_sigma_Y["mean"], "sigma": mu_sigma_Y["std"], "count": mu_sigma_Y["count"]},
        },
        "diagnostics": diag_stats,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2)

    print(f"\n[OK] Wrote: {OUT_JSON.resolve()}")
    print("[KEY]")
    print(f"  T0_fixed = {T0_FIXED:.6e}")
    print(f"  q0       = {q0:.6e}")
    print(f"  y0       = {y0:.6e}")
    print(f"  mu/sigma(X_T) = {mu_sigma_XT['mean']:.6e} / {mu_sigma_XT['std']:.6e}")
    print(f"  mu/sigma(X_q) = {mu_sigma_Xq['mean']:.6e} / {mu_sigma_Xq['std']:.6e}")
    print(f"  mu/sigma(Y)   = {mu_sigma_Y['mean']:.6e} / {mu_sigma_Y['std']:.6e}")


if __name__ == "__main__":
    main()
