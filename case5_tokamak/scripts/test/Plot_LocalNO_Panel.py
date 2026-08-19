#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot_LocalNO_Revised.py

4 geometries (CMOD_AT_, CMOD_NewGeo_SO_, SPARC_, ARC_) — excludes CMOD_SO_.
For each geometry & split, scan ALL samples, compute masked MSE on the
normalized prediction (disk domain), then pick the lowest-loss and
highest-loss sample.

Produces 4 figures (each 4 rows × 4 cols, layout identical to Test.py):
    train_lowest_loss.png
    train_highest_loss.png
    val_lowest_loss.png
    val_highest_loss.png

Columns:
    0  Unit-disk delta T     (RdBu_r, 1:1 aspect)
    1  Physical pred Te      (inferno, H:W = 2:1)
    2  Physical bench Te     (inferno, H:W = 2:1, shared cbar with col 1)
    3  Physical |error|      (hot_r,   H:W = 2:1)

Harmonic-mapping meshes are loaded/precomputed once per geometry.

Usage:
    python Plot_LocalNO_Revised.py
"""

from __future__ import annotations

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata

from Data import load_stats, normalize_ic, normalize_source
from Model_LocalNO_L_2 import SingleLocalNO


# ======================================================================
# CONFIGURATION
# ======================================================================

CONFIG = {
    # ---- data & model -------------------------------------------------
    "data_h5":    Path("Data/Merge/Data_ML_Merged.h5"),
    "stats_json": Path("Data/CalStat/stats_train.json"),
    "model_pt":   Path("Train/LocalNO_Large/Train_2/best_model.pt"),

    # ---- 4 geometries (exclude CMOD_SO_) ------------------------------
    "prefixes": ["CMOD_AT_", "CMOD_NewGeo_SO_", "SPARC_", "ARC_"],

    # ---- output -------------------------------------------------------
    "output_dir":  Path("plot/plot_LocalNO_L_panel"),
    "dpi":         400,

    # ---- device -------------------------------------------------------
    "device": "cuda",

    # ---- physical-domain reconstruction -------------------------------
    "phys_res":        512,
    "phys_pad_frac":   0.01,
    "disk_group_256":  "disk_256",

    # ---- harmonic-mapping H5 per prefix -------------------------------
    "hm_paths": {
        "CMOD_AT_":        Path("HM/CMOD_HM.h5"),
        "CMOD_NewGeo_SO_": Path("HM/CMOD_NewGeo_HM.h5"),
        "SPARC_":          Path("HM/SPARC_HM.h5"),
        "ARC_":            Path("HM/ARC_V02_HM.h5"),
    },

    # ---- masking ------------------------------------------------------
    "mask_thr": 0.5,

    # ---- absolute-error floor -----------------------------------------
    "abs_err_floor": 1e-5,
}


# ======================================================================
# LOGGING
# ======================================================================

_LOG_FH = None


def init_log(output_dir: Path):
    global _LOG_FH
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"Plot_{ts}.log"
    _LOG_FH = open(log_path, "w", encoding="utf-8")
    log(f"[LOG] Logging to {log_path.resolve()}")


def log(msg: str = ""):
    print(msg, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()


def close_log():
    global _LOG_FH
    if _LOG_FH is not None:
        _LOG_FH.close()
        _LOG_FH = None


# ======================================================================
# DENORMALIZATION
# ======================================================================

def denormalize_label(y_norm: np.ndarray, stats: Dict) -> np.ndarray:
    mu = stats["mu_sigma"]["Y"]["mu"]
    sigma = stats["mu_sigma"]["Y"]["sigma"]
    y0 = stats["scales"]["y0"]
    return np.sinh(np.asarray(y_norm, dtype=np.float64) * sigma + mu) * y0


# ======================================================================
# HARMONIC-MAPPING HELPERS  (256 only)
# ======================================================================

def load_hm_mapping(hm_path: Path, grp256: str) -> Dict[str, Any]:
    with h5py.File(hm_path, "r") as f:
        pts = np.asarray(f["mesh/pts"][...], dtype=np.float64)
        tri = np.asarray(f["mesh/tri"][...], dtype=np.int64)
        boundary_nodes = np.asarray(f["mesh/boundary_nodes"][...], dtype=np.int64)

        g = f[f"disk/{grp256}"]
        d256 = {
            "xd":         np.asarray(g["xd"][...], dtype=np.float64),
            "yd":         np.asarray(g["yd"][...], dtype=np.float64),
            "inv_tri_id": np.asarray(g["inv_tri_id"][...], dtype=np.int32),
            "inv_bary":   np.asarray(g["inv_bary"][...], dtype=np.float32),
            "valid_mask": np.asarray(g["valid_mask"][...], dtype=np.uint8),
        }

    return {
        "pts": pts,
        "tri": tri,
        "boundary_nodes": boundary_nodes,
        "d256": d256,
    }


def map_back_disk_to_physical(
    pts, tri, inv_tri_id, inv_bary, valid_mask,
) -> Tuple[np.ndarray, np.ndarray]:
    Ny, Nx = inv_tri_id.shape
    Rb = np.full((Ny, Nx), np.nan, dtype=np.float64)
    Zb = np.full((Ny, Nx), np.nan, dtype=np.float64)
    ok = valid_mask.astype(bool)
    if not np.any(ok):
        return Rb, Zb

    t = inv_tri_id[ok].astype(np.int64)
    lam = inv_bary[ok].astype(np.float64)
    a, b, c = tri[t, 0], tri[t, 1], tri[t, 2]

    Rb[ok] = lam[:, 0] * pts[a, 0] + lam[:, 1] * pts[b, 0] + lam[:, 2] * pts[c, 0]
    Zb[ok] = lam[:, 0] * pts[a, 1] + lam[:, 1] * pts[b, 1] + lam[:, 2] * pts[c, 1]
    return Rb, Zb


def precompute_physical_grid(hm: Dict, cfg: Dict) -> Dict[str, Any]:
    d256 = hm["d256"]
    pts, tri, bnodes = hm["pts"], hm["tri"], hm["boundary_nodes"]

    Rb, Zb = map_back_disk_to_physical(
        pts, tri, d256["inv_tri_id"], d256["inv_bary"], d256["valid_mask"],
    )
    ok_geo = d256["valid_mask"].astype(bool) & np.isfinite(Rb) & np.isfinite(Zb)

    pad = cfg["phys_pad_frac"]
    rmin, rmax = float(Rb[ok_geo].min()), float(Rb[ok_geo].max())
    zmin, zmax = float(Zb[ok_geo].min()), float(Zb[ok_geo].max())
    pr = pad * (rmax - rmin + 1e-12)
    pz = pad * (zmax - zmin + 1e-12)

    res = cfg["phys_res"]
    R_grid = np.linspace(rmin - pr, rmax + pr, res)
    Z_grid = np.linspace(zmin - pz, zmax + pz, res)
    RR, ZZ = np.meshgrid(R_grid, Z_grid, indexing="xy")

    poly = MplPath(pts[bnodes, :2], closed=True)
    inside = poly.contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)

    bphys = pts[bnodes, :2]

    return {
        "Rb": Rb,
        "Zb": Zb,
        "ok_geo": ok_geo,
        "R_grid": R_grid,
        "Z_grid": Z_grid,
        "RR": RR,
        "ZZ": ZZ,
        "inside": inside,
        "bphys": bphys,
    }


def field_to_physical(field256: np.ndarray, hm: Dict, phys_pre: Dict) -> np.ndarray:
    ok = phys_pre["ok_geo"] & np.isfinite(field256)
    Rb, Zb = phys_pre["Rb"], phys_pre["Zb"]
    R_sc, Z_sc, V_sc = Rb[ok], Zb[ok], field256[ok]

    bphys = phys_pre["bphys"]
    V_bnd = griddata(np.c_[R_sc, Z_sc], V_sc, bphys, method="nearest")
    pts_all = np.vstack([np.c_[R_sc, Z_sc], bphys])
    vals_all = np.concatenate([V_sc, V_bnd])

    field_phys = griddata(
        pts_all,
        vals_all,
        (phys_pre["RR"], phys_pre["ZZ"]),
        method="linear"
    )
    field_phys[~phys_pre["inside"]] = np.nan
    return field_phys


# ======================================================================
# DATA LOADING & INFERENCE
# ======================================================================

def build_prefix_split_indices(stats: Dict, data_h5: Path, prefixes: List[str]):
    """Return {pfx: {'train': [gidx, ...], 'val': [gidx, ...]}} sorted."""
    split = stats["split"]
    train_set = set(np.array(split["train_idx"], dtype=np.int64).tolist())
    val_set = set(np.array(split["val_idx"], dtype=np.int64).tolist())

    with h5py.File(data_h5, "r") as f:
        case_ids = [
            (x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x))
            for x in np.asarray(f["case_id"][...])
        ]

    out = {}
    for pfx in prefixes:
        pfx_train = sorted(
            i for i in range(len(case_ids))
            if case_ids[i].startswith(pfx) and i in train_set
        )
        pfx_val = sorted(
            i for i in range(len(case_ids))
            if case_ids[i].startswith(pfx) and i in val_set
        )
        out[pfx] = {"train": pfx_train, "val": pfx_val}
        log(f"  {pfx:<20s}  train={len(pfx_train):>5d}   val={len(pfx_val):>5d}")

    return out, case_ids


def load_raw_sample(h5_path, idx):
    with h5py.File(h5_path, "r") as f:
        ic = np.nan_to_num(np.asarray(f["IC"][idx], dtype=np.float64))
        src = np.nan_to_num(np.asarray(f["Source"][idx], dtype=np.float64))
        label = np.nan_to_num(np.asarray(f["Label"][idx], dtype=np.float64))
        bench = np.nan_to_num(np.asarray(f["Benchmark"][idx], dtype=np.float64))
    return {"IC": ic, "Source": src, "Label": label, "Benchmark": bench}


def run_inference_normed(model, device, stats, raw, mask):
    """
    Run inference and return both normalized prediction and mask for MSE,
    plus denormalized fields for plotting.
    """
    ic_norm = normalize_ic(raw["IC"], stats)
    src_norm = normalize_source(raw["Source"], stats)
    x_np = np.stack([ic_norm, src_norm], axis=0).astype(np.float32)

    x_gpu = torch.from_numpy(x_np).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_gpu = model(x_gpu)
    pred_norm = pred_gpu.cpu().numpy().astype(np.float64)[0, 0]  # (H, W)

    mu = stats["mu_sigma"]["Y"]["mu"]
    sigma = stats["mu_sigma"]["Y"]["sigma"]
    y0 = stats["scales"]["y0"]
    label_norm = (np.arcsinh(raw["Label"].astype(np.float64) / y0) - mu) / sigma

    m = mask.astype(bool)
    mse = float(np.mean((pred_norm[m] - label_norm[m]) ** 2))

    dT_pred = denormalize_label(pred_norm, stats)
    dT_true = raw["Label"].copy()
    T_pred_disk = raw["IC"] + dT_pred
    T_true_disk = raw["Benchmark"].copy()

    return mse, {
        "dT_pred": dT_pred,
        "dT_true": dT_true,
        "T_pred_disk": T_pred_disk,
        "T_true_disk": T_true_disk,
    }


def apply_disk_mask(field, mask, thr=0.5):
    out = np.array(field, dtype=np.float64, copy=True)
    if mask is not None:
        out[mask < thr] = np.nan
    return out


# ======================================================================
# SCAN ALL SAMPLES → PICK LOWEST / HIGHEST LOSS
# ======================================================================

def scan_prefix_split(
    model, device, stats, data_h5: Path, indices: List[int], mask: np.ndarray,
) -> List[Tuple[int, float]]:
    """Return list of (global_idx, mse) for every sample in the given index list."""
    results = []
    for gidx in indices:
        raw = load_raw_sample(data_h5, gidx)
        mse, _ = run_inference_normed(model, device, stats, raw, mask)
        results.append((gidx, mse))
    return results


# ======================================================================
# COLORBAR HELPERS
# ======================================================================

def _format_sig2(val: float) -> str:
    if np.isclose(val, 0.0):
        return "0"
    return f"{val:.2g}"


def _sci_cbar(fig, mappable, cax, vmin, vmax, n_tick, fs):
    """
    Add a colorbar with uniform scientific-notation tick labels.
    Tick labels are shown after scaling by 10^exp and formatted with
    two significant digits.
    """
    cb = fig.colorbar(mappable, cax=cax)
    cb.ax.tick_params(labelsize=fs)

    span = max(abs(vmin), abs(vmax))
    if span > 0:
        exp = int(np.floor(np.log10(span)))
    else:
        exp = 0

    scale = 10.0 ** exp
    raw_ticks = np.linspace(vmin, vmax, n_tick)
    scaled_ticks = raw_ticks / scale

    cb.set_ticks(raw_ticks)
    cb.set_ticklabels([_format_sig2(t) for t in scaled_ticks])
    cb.ax.set_title(f"$\\times 10^{{{exp}}}$", fontsize=fs, pad=4)
    return cb


# ======================================================================
# PLOTTING  (Test.py layout, 4 rows × 4 cols)
# ======================================================================

def plot_combined_figure(
    fig_label: str,
    row_results: List[Dict],
    out_path: Path,
    dpi: int,
):
    """
    row_results: list of 4 dicts, one per geometry row, each containing:
        label, pred_dT_disk, Te_pred_phys, Te_bench_phys, abs_err_phys,
        R, Z
    Layout identical to Test.py.
    """
    n_rows = len(row_results)
    panel_h = 3.0
    disk_w = panel_h * 1.0   # 1:1
    phys_w = panel_h * 0.5   # H:W = 2:1

    gap_01 = 0.70
    gap_12 = 0.15
    gap_23 = 0.70
    cbar_w = 0.12
    cbar_pad = 0.08

    hspace_in = 0.40
    left_m, right_m = 0.05, 0.05
    top_m, bot_m = 0.35, 0.10

    total_w = (
        left_m + disk_w + cbar_pad + cbar_w
        + gap_01 + phys_w + gap_12 + phys_w + cbar_pad + cbar_w
        + gap_23 + phys_w + cbar_pad + cbar_w + right_m
    )
    total_h = top_m + n_rows * panel_h + (n_rows - 1) * hspace_in + bot_m

    fig = plt.figure(figsize=(total_w, total_h), dpi=dpi)

    def nx(inches): return inches / total_w
    def ny(inches): return inches / total_h

    pw = nx(phys_w)
    dw = nx(disk_w)
    cbw = nx(cbar_w)
    cbp = nx(cbar_pad)

    x0 = nx(left_m)
    x0_cb = x0 + dw + cbp
    x1 = x0_cb + cbw + nx(gap_01)
    x2 = x1 + pw + nx(gap_12)
    x2_cb = x2 + pw + cbp
    x3 = x2_cb + cbw + nx(gap_23)
    x3_cb = x3 + pw + cbp

    ph = ny(panel_h)
    hs = ny(hspace_in)
    y_bot = ny(bot_m)

    fs_tick = 9
    fs_cbar = 9
    cmap_te = "inferno"
    cmap_err = "hot_r"
    cmap_dt = "RdBu_r"

    N_XTICK_DISK = 5
    N_YTICK_DISK = 5
    N_XTICK_PHYS = 4
    N_YTICK_PHYS = 6
    N_CBAR_TICK = 6

    for row, res in enumerate(row_results):
        y = y_bot + (n_rows - 1 - row) * (ph + hs)

        R, Z = res["R"], res["Z"]
        extent_phys = [R.min(), R.max(), Z.min(), Z.max()]

        # ---- Col 0: disk delta T (same style as Test.py) ----
        ax0 = fig.add_axes([x0, y, dw, ph])
        dT = res["pred_dT_disk"]
        vabs = np.nanmax(np.abs(dT))
        if (not np.isfinite(vabs)) or vabs == 0:
            vabs = 1.0

        im0 = ax0.imshow(
            dT,
            origin="lower",
            aspect="equal",
            extent=[-1, 1, -1, 1],
            cmap=cmap_dt,
            vmin=-vabs,
            vmax=vabs,
        )
        ax0.set_xlim(-1, 1)
        ax0.set_ylim(-1, 1)
        ax0.xaxis.set_major_locator(MultipleLocator(0.5))
        ax0.yaxis.set_major_locator(MultipleLocator(0.5))
        ax0.tick_params(labelsize=fs_tick)

        # Col 0 colorbar — same style as Test.py
        cax0 = fig.add_axes([x0_cb, y, cbw, ph])
        cb0 = fig.colorbar(im0, cax=cax0)
        cb0.ax.tick_params(labelsize=fs_cbar)

        if vabs > 0:
            exp = int(np.floor(np.log10(vabs)))
        else:
            exp = 0

        scale = 10.0 ** exp
        raw_ticks = np.linspace(-vabs, vabs, N_CBAR_TICK)
        cb0.set_ticks(raw_ticks)
        cb0.set_ticklabels([f"{t / scale:.1f}" for t in raw_ticks])
        cb0.ax.set_title(f"$\\times 10^{{{exp}}}$", fontsize=fs_cbar, pad=4)

        # ---- shared vmin/vmax for Col 1 & 2 ----
        vmin_te = np.nanmin([
            np.nanmin(res["Te_pred_phys"]),
            np.nanmin(res["Te_bench_phys"]),
        ])
        vmax_te = np.nanmax([
            np.nanmax(res["Te_pred_phys"]),
            np.nanmax(res["Te_bench_phys"]),
        ])

        # ---- Col 1: physical pred Te ----
        ax1 = fig.add_axes([x1, y, pw, ph])
        im1 = ax1.imshow(
            res["Te_pred_phys"],
            origin="lower",
            aspect="equal",
            extent=extent_phys,
            cmap=cmap_te,
            vmin=vmin_te,
            vmax=vmax_te,
        )
        ax1.xaxis.set_major_locator(MaxNLocator(N_XTICK_PHYS))
        ax1.yaxis.set_major_locator(MaxNLocator(N_YTICK_PHYS))
        ax1.tick_params(labelsize=fs_tick)

        # ---- Col 2: physical bench Te ----
        ax2 = fig.add_axes([x2, y, pw, ph])
        im2 = ax2.imshow(
            res["Te_bench_phys"],
            origin="lower",
            aspect="equal",
            extent=extent_phys,
            cmap=cmap_te,
            vmin=vmin_te,
            vmax=vmax_te,
        )
        ax2.xaxis.set_major_locator(MaxNLocator(N_XTICK_PHYS))
        ax2.yaxis.set_major_locator(MaxNLocator(N_YTICK_PHYS))
        ax2.tick_params(labelsize=fs_tick)

        # Shared colorbar for Col 1 & 2
        cax12 = fig.add_axes([x2_cb, y, cbw, ph])
        cb12 = fig.colorbar(im2, cax=cax12)
        cb12.locator = MaxNLocator(N_CBAR_TICK)
        cb12.update_ticks()
        cb12.ax.tick_params(labelsize=fs_cbar)

        # ---- Col 3: physical |error| ----
        ax3 = fig.add_axes([x3, y, pw, ph])
        err_vmin = 0.0
        err_valid = res["abs_err_phys"]
        err_vmax = float(np.nanmax(err_valid)) if np.any(np.isfinite(err_valid)) else 1.0
        if (not np.isfinite(err_vmax)) or err_vmax <= 0:
            err_vmax = 1.0

        im3 = ax3.imshow(
            res["abs_err_phys"],
            origin="lower",
            aspect="equal",
            extent=extent_phys,
            cmap=cmap_err,
            vmin=err_vmin,
            vmax=err_vmax,
        )
        ax3.xaxis.set_major_locator(MaxNLocator(N_XTICK_PHYS))
        ax3.yaxis.set_major_locator(MaxNLocator(N_YTICK_PHYS))
        ax3.tick_params(labelsize=fs_tick)

        # Col 3 colorbar — scientific notation
        cax3 = fig.add_axes([x3_cb, y, cbw, ph])
        _sci_cbar(fig, im3, cax3, err_vmin, err_vmax, N_CBAR_TICK, fs_cbar)

    # fig.suptitle(fig_label, fontsize=13, y=1.0 - ny(top_m) * 0.35)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    log(f"  -> Saved {out_path}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    cfg = CONFIG
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    init_log(out_dir)

    device = torch.device(cfg["device"])
    stats = load_stats(cfg["stats_json"])

    # --- load mask ---
    with h5py.File(cfg["data_h5"], "r") as f:
        mask = np.asarray(f["Mask"], dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[0]

    # --- build per-prefix index lists ---
    log("[INFO] Building per-prefix sample lists ...")
    pfx_splits, case_ids = build_prefix_split_indices(
        stats, cfg["data_h5"], cfg["prefixes"],
    )

    # --- load model ---
    model = SingleLocalNO().to(device)
    ckpt = torch.load(cfg["model_pt"], map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k: v for k, v in state.items() if not k.startswith("_")}
    model.load_state_dict(state)
    model.eval()
    log(f"[INFO] Model loaded from {cfg['model_pt']}")

    # --- precompute HM & physical grids (once per geometry) ---
    hm_cache: Dict[str, Any] = {}
    phys_cache: Dict[str, Any] = {}

    for pfx in cfg["prefixes"]:
        hm_path = cfg["hm_paths"][pfx]
        if not hm_path.exists():
            log(f"  [WARN] HM file missing for {pfx}; skipping")
            continue
        hm = load_hm_mapping(hm_path, cfg["disk_group_256"])
        hm_cache[pfx] = hm
        phys_cache[pfx] = precompute_physical_grid(hm, cfg)
        log(f"  [INFO] HM & physical grid ready for {pfx}")

    # --- scan all samples per prefix per split → rank by loss ---
    picks: Dict[str, Dict[str, Dict[str, int]]] = {}

    for pfx in cfg["prefixes"]:
        if pfx not in hm_cache:
            continue
        picks[pfx] = {}
        for split_name in ("train", "val"):
            indices = pfx_splits[pfx][split_name]
            n = len(indices)
            log(f"\n[SCAN] {pfx} {split_name}: {n} samples ...")

            losses = scan_prefix_split(
                model, device, stats, cfg["data_h5"], indices, mask,
            )
            losses.sort(key=lambda x: x[1])

            gidx_low, mse_low = losses[0]
            gidx_high, mse_high = losses[-1]

            log(
                f"  lowest  loss: gidx={gidx_low:>5d}  "
                f"({case_ids[gidx_low]})  MSE={mse_low:.6e}"
            )
            log(
                f"  highest loss: gidx={gidx_high:>5d}  "
                f"({case_ids[gidx_high]})  MSE={mse_high:.6e}"
            )

            picks[pfx][split_name] = {
                "lowest": gidx_low,
                "highest": gidx_high,
                "mse_lowest": mse_low,
                "mse_highest": mse_high,
            }

    # --- produce 4 figures ---
    figure_specs = [
        ("train", "lowest",  "Train — Lowest Loss"),
        ("train", "highest", "Train — Highest Loss"),
        ("val",   "lowest",  "Validation — Lowest Loss"),
        ("val",   "highest", "Validation — Highest Loss"),
    ]

    for split_name, rank_key, fig_title in figure_specs:
        log(f"\n{'=' * 60}")
        log(f"[PLOT] {fig_title}")
        log(f"{'=' * 60}")

        row_results = []

        for pfx in cfg["prefixes"]:
            if pfx not in picks or split_name not in picks[pfx]:
                continue

            gidx = picks[pfx][split_name][rank_key]
            mse_val = picks[pfx][split_name][f"mse_{rank_key}"]
            pfx_clean = pfx.rstrip("_")

            log(
                f"  {pfx_clean}: gidx={gidx} ({case_ids[gidx]})  "
                f"MSE={mse_val:.6e}"
            )

            raw = load_raw_sample(cfg["data_h5"], gidx)
            _, res = run_inference_normed(model, device, stats, raw, mask)

            hm = hm_cache[pfx]
            phys_pre = phys_cache[pfx]

            # disk delta T masked
            pred_dT_disk = apply_disk_mask(res["dT_pred"], mask, cfg["mask_thr"])

            # physical Te
            Te_pred_phys = field_to_physical(res["T_pred_disk"], hm, phys_pre)
            Te_bench_phys = field_to_physical(res["T_true_disk"], hm, phys_pre)

            # absolute error (zero where |bench| < floor)
            abs_err = np.full_like(Te_bench_phys, np.nan)
            valid_phys = np.isfinite(Te_bench_phys) & np.isfinite(Te_pred_phys)
            abs_err[valid_phys] = np.abs(
                Te_pred_phys[valid_phys] - Te_bench_phys[valid_phys]
            )
            small = np.abs(Te_bench_phys) < cfg["abs_err_floor"]
            abs_err[valid_phys & small] = 0.0

            row_results.append(dict(
                label=pfx_clean,
                pred_dT_disk=pred_dT_disk,
                Te_pred_phys=Te_pred_phys,
                Te_bench_phys=Te_bench_phys,
                abs_err_phys=abs_err,
                R=phys_pre["R_grid"],
                Z=phys_pre["Z_grid"],
            ))

        fname = f"{split_name}_{rank_key}_loss.png"
        plot_combined_figure(
            fig_title,
            row_results,
            out_dir / fname,
            cfg["dpi"],
        )

    log(f"\n[DONE] All outputs in {out_dir.resolve()}")
    close_log()


if __name__ == "__main__":
    main()