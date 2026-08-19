#!/usr/bin/env python3
"""
Plot_GPU_Aitken_results_revised.py

Read the GPU/Aitken Pe-Da scan HDF5 file and generate figures only.

Figure 1:
  - Keep the original summary layout:
      central smooth 2D contour of mean_relL2
      top marginal: mean relL2 vs Pe averaged over Da
      right marginal: mean relL2 vs Da averaged over Pe
      horizontal colorbar below central contour
  - Only change the x/y ticks of the central contour to [1, 5, 10, 15, 20]
    when those tick values fall inside the scan range.

Figure 2:
  - 2D contour of mean ||S_eff||_2 / ||S||_2
  - Same Pe/Da limits, ticks, and aspect ratio as the central panel of Figure 1.
  - Mark canonical point.

Figure 3:
  - 2D contour of mean iteration count
  - Same Pe/Da limits, ticks, and aspect ratio as the central panel of Figure 1.
  - Mark canonical point.

Required HDF5 datasets:
  - Pe_list
  - Da_list
  - mean_relL2
  - std_relL2
  - mean_Seff_ratio
  - mean_iters

Expected array shape:
  - mean_relL2, mean_Seff_ratio, mean_iters: (nDa, nPe)
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.interpolate import RegularGridInterpolator


# -----------------------------
# User settings
# -----------------------------
IN_H5 = "peda_scan_bandlimit_results.h5"

OUT_PNG_SUMMARY = "peda_scan_bandlimit_summary_mean_gpu_aitken.png"
OUT_PNG_SEFF    = "peda_scan_bandlimit_mean_Seff_ratio_contour_gpu_aitken.png"
OUT_PNG_ITERS   = "peda_scan_bandlimit_mean_iters_contour_gpu_aitken.png"

PLOT_RES = 0.1
N_LEVELS = 50

# Requested central-panel axis ticks.
REQUESTED_TICKS = np.array([1, 5, 10, 15, 20], dtype=np.float64)

# --- Red contour overlay: on/off switch and threshold for Figure 1 only ---
SHOW_ERROR_CONTOUR = True
ERROR_THRESHOLD = 0.01

# --- Style controls: keep these from the original summary figure ---
FIGSIZE = (14, 16)

FS_MAIN_LABEL = 16
FS_MAIN_TICK = 14
FS_TITLE = 16
FS_MARG_LABEL = 16
FS_MARG_TICK = 12
FS_CBAR_LABEL = 12
FS_CBAR_TICK = 10

STAR_SIZE = 20
STAR_EDGEWIDTH = 0.9

TOP_LINEWIDTH = 1.8
RIGHT_LINEWIDTH = 1.8
MARG_MARKERSIZE = 4.0

# --- Standalone contour figures: same central-panel visual proportion as Figure 1 ---
# The central panel ratio in the original GridSpec is approximately width:height = 5.2:5.0.
FIGSIZE_CONTOUR = (8.3, 8.0)


# -----------------------------
# Formatting helpers
# -----------------------------
def compact_sci_formatter(x, pos=None):
    """
    Compact formatter:
      - integers / simple decimals if clean enough
      - scientific notation for very small / very large values
    """
    if not np.isfinite(x):
        return ""

    if x == 0:
        return "0"

    ax = abs(x)
    if ax >= 1e3 or ax < 1e-3:
        return f"{x:.0e}"
    elif ax < 1e-2:
        return f"{x:.1e}"
    elif ax < 1:
        return f"{x:.3f}".rstrip("0").rstrip(".")
    else:
        return f"{x:.2f}".rstrip("0").rstrip(".")


COMPACT = FuncFormatter(compact_sci_formatter)


def nanmean_checked(a: np.ndarray, axis: int) -> np.ndarray:
    """
    np.nanmean emits warnings for all-NaN slices. This helper returns NaN for
    all-NaN slices without warnings.
    """
    a = np.asarray(a, dtype=np.float64)
    finite = np.isfinite(a)
    count = np.sum(finite, axis=axis)
    total = np.sum(np.where(finite, a, 0.0), axis=axis)
    out = np.full(count.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


def require_dataset(h5: h5py.File, name: str) -> np.ndarray:
    if name not in h5:
        available = ", ".join(sorted(h5.keys()))
        raise KeyError(f"Dataset '{name}' not found in {IN_H5}. Available datasets: {available}")
    return h5[name][...]


def validate_grid_shape(name: str, arr: np.ndarray, nDa: int, nPe: int) -> None:
    if arr.shape != (nDa, nPe):
        raise ValueError(
            f"Dataset '{name}' has shape {arr.shape}, expected {(nDa, nPe)} "
            f"for arrays indexed as (Da, Pe)."
        )


def axis_ticks_within_range(vmin: float, vmax: float) -> np.ndarray:
    ticks = REQUESTED_TICKS[(REQUESTED_TICKS >= vmin) & (REQUESTED_TICKS <= vmax)]
    if ticks.size > 0:
        return ticks
    return np.linspace(vmin, vmax, 5)


def interpolate_to_fine_grid(
    Pe_list: np.ndarray,
    Da_list: np.ndarray,
    Z: np.ndarray,
    Pe_min: float,
    Pe_max: float,
    Da_min: float,
    Da_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate Z(Da, Pe) to a fine regular grid.
    """
    rgi = RegularGridInterpolator(
        (Da_list, Pe_list), Z,
        method="linear", bounds_error=False, fill_value=None,
    )

    pe_f = np.linspace(Pe_min, Pe_max, int(round((Pe_max - Pe_min) / PLOT_RES)) + 1)
    da_f = np.linspace(Da_min, Da_max, int(round((Da_max - Da_min) / PLOT_RES)) + 1)
    PE_F, DA_F = np.meshgrid(pe_f, da_f)

    pts = np.stack([DA_F.ravel(), PE_F.ravel()], axis=-1)
    ZF_data = rgi(pts).reshape(PE_F.shape)

    return PE_F, DA_F, ZF_data


def finite_minmax(Z: np.ndarray, name: str) -> tuple[float, float]:
    finite_vals = Z[np.isfinite(Z)]
    if finite_vals.size == 0:
        raise RuntimeError(f"No finite values for {name}.")

    vmin = float(np.nanmin(finite_vals))
    vmax = float(np.nanmax(finite_vals))

    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 1e-6, 1e-12)
        vmin -= delta
        vmax += delta

    return vmin, vmax


# -----------------------------
# Figure 1: original summary plot, only central axis ticks changed
# -----------------------------
def plot_summary_mean_relL2(
    Pe_list: np.ndarray,
    Da_list: np.ndarray,
    mean_relL2: np.ndarray,
    std_relL2: np.ndarray,
    Pe_star: float,
    Da_star: float,
    Pe_min: float,
    Pe_max: float,
    Da_min: float,
    Da_max: float,
) -> None:
    # Marginal statistics
    mean_pe = nanmean_checked(mean_relL2, axis=0)   # average over Da -> function of Pe
    mean_da = nanmean_checked(mean_relL2, axis=1)   # average over Pe -> function of Da

    PE_F, DA_F, ZF_data = interpolate_to_fine_grid(
        Pe_list, Da_list, mean_relL2, Pe_min, Pe_max, Da_min, Da_max
    )

    vmin, vmax = finite_minmax(ZF_data, "mean_relL2 contour")
    levels = np.linspace(vmin, vmax, N_LEVELS)

    plt.rcParams.update({
        "font.size": FS_MAIN_TICK,
        "axes.labelsize": FS_MAIN_LABEL,
        "axes.titlesize": FS_TITLE,
        "axes.labelweight": "bold",
        "xtick.labelsize": FS_MAIN_TICK,
        "ytick.labelsize": FS_MAIN_TICK,
    })

    # KEEP original figure layout/proportions.
    fig = plt.figure(figsize=FIGSIZE)

    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[5.2, 1.3],
        height_ratios=[0.85, 5.0],
        hspace=0.06,
        wspace=0.06,
        left=0.11, right=0.92, bottom=0.15, top=0.95,
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis("off")

    # Central contour panel
    norm = Normalize(vmin=vmin, vmax=vmax)
    cf = ax_main.contourf(
        PE_F, DA_F, ZF_data,
        levels=levels,
        norm=norm,
        cmap="viridis",
        extend="neither",
    )

    if SHOW_ERROR_CONTOUR:
        if vmin <= ERROR_THRESHOLD <= vmax:
            ax_main.contour(
                PE_F, DA_F, ZF_data,
                levels=[ERROR_THRESHOLD],
                colors="red",
                linewidths=2.2,
                linestyles="--",
                zorder=4,
            )
        else:
            print(
                f"[Info] Skipping error contour at {ERROR_THRESHOLD:g}: "
                f"outside data range [{vmin:g}, {vmax:g}]."
            )

    ax_main.plot(
        [Pe_star], [Da_star], "*",
        color="red",
        markersize=STAR_SIZE,
        markeredgecolor="k",
        markeredgewidth=STAR_EDGEWIDTH,
        zorder=5,
    )

    ax_main.set_xlim(Pe_min, Pe_max)
    ax_main.set_ylim(Da_min, Da_max)

    # ONLY requested change for Figure 1 central panel axes.
    ax_main.set_xticks(axis_ticks_within_range(Pe_min, Pe_max))
    ax_main.set_yticks(axis_ticks_within_range(Da_min, Da_max))

    ax_main.set_xlabel("Pe", fontweight="bold", fontsize=FS_MAIN_LABEL)
    ax_main.set_ylabel("Da", fontweight="bold", fontsize=FS_MAIN_LABEL)
    ax_main.tick_params(axis="both", labelsize=FS_MAIN_TICK, width=1.0, length=5)
    ax_main.grid(True, alpha=0.18, color="w", linewidth=0.6)

    # Colorbar — original style
    fig.canvas.draw()
    bbox_main = ax_main.get_position()
    cbar_ax = fig.add_axes([
        bbox_main.x0,
        bbox_main.y0 - 0.075,
        bbox_main.width,
        0.018
    ])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Mean Relative L2", fontsize=FS_CBAR_LABEL, fontweight="bold")
    cbar.ax.tick_params(labelsize=FS_CBAR_TICK)

    cbar_ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.xaxis.set_major_formatter(COMPACT)

    # Top marginal — original
    color_pe = "#2166ac"
    ax_top.plot(
        Pe_list, mean_pe, "-o",
        color=color_pe,
        markersize=MARG_MARKERSIZE,
        linewidth=TOP_LINEWIDTH,
    )
    ax_top.set_xlim(Pe_min, Pe_max)
    ax_top.set_ylabel("Mean Rel. L2", fontsize=FS_MARG_LABEL, fontweight="bold", labelpad=2)
    ax_top.yaxis.set_major_formatter(COMPACT)
    ax_top.yaxis.set_major_locator(MaxNLocator(nbins=2))
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=FS_MARG_TICK)
    ax_top.grid(True, alpha=0.22)
    ax_top.set_title("Marginal over Da", fontsize=FS_MARG_LABEL + 1, pad=4)

    # Right marginal — original
    color_da = "#b2182b"
    ax_right.plot(
        mean_da, Da_list, "-o",
        color=color_da,
        markersize=MARG_MARKERSIZE,
        linewidth=RIGHT_LINEWIDTH,
    )
    ax_right.set_ylim(Da_min, Da_max)
    ax_right.set_xlabel("Mean Rel. L2", fontsize=FS_MARG_LABEL, fontweight="bold", labelpad=2)
    ax_right.xaxis.set_major_formatter(COMPACT)
    ax_right.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.tick_params(axis="x", labelsize=FS_MARG_TICK, rotation=35)
    ax_right.grid(True, alpha=0.22)
    ax_right.set_title("Marginal\nover Pe", fontsize=FS_MARG_LABEL + 1, pad=4)

    for ax in [ax_main, ax_top, ax_right]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    fig.savefig(OUT_PNG_SUMMARY, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print(f"[Done] Saved figure: {OUT_PNG_SUMMARY}")


# -----------------------------
# Figures 2-3: standalone contours with same Pe-Da axes/proportion as Figure 1 central panel
# -----------------------------
def plot_standalone_contour(
    Pe_list: np.ndarray,
    Da_list: np.ndarray,
    Z: np.ndarray,
    out_png: str,
    *,
    title: str,
    cbar_label: str,
    Pe_star: float,
    Da_star: float,
    Pe_min: float,
    Pe_max: float,
    Da_min: float,
    Da_max: float,
) -> None:
    PE_F, DA_F, ZF_data = interpolate_to_fine_grid(
        Pe_list, Da_list, Z, Pe_min, Pe_max, Da_min, Da_max
    )

    vmin, vmax = finite_minmax(ZF_data, title)
    levels = np.linspace(vmin, vmax, N_LEVELS)

    plt.rcParams.update({
        "font.size": FS_MAIN_TICK,
        "axes.labelsize": FS_MAIN_LABEL,
        "axes.titlesize": FS_TITLE,
        "axes.labelweight": "bold",
        "xtick.labelsize": FS_MAIN_TICK,
        "ytick.labelsize": FS_MAIN_TICK,
    })

    # Match the central-panel visual proportion, not the full marginal figure.
    fig, ax = plt.subplots(figsize=FIGSIZE_CONTOUR)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cf = ax.contourf(
        PE_F, DA_F, ZF_data,
        levels=levels,
        norm=norm,
        cmap="viridis",
        extend="neither",
    )

    ax.plot(
        [Pe_star], [Da_star], "*",
        color="red",
        markersize=STAR_SIZE,
        markeredgecolor="k",
        markeredgewidth=STAR_EDGEWIDTH,
        zorder=5,
    )

    ax.set_xlim(Pe_min, Pe_max)
    ax.set_ylim(Da_min, Da_max)

    # Same ticks as Figure 1 central panel.
    ax.set_xticks(axis_ticks_within_range(Pe_min, Pe_max))
    ax.set_yticks(axis_ticks_within_range(Da_min, Da_max))

    # Preserve the same data aspect behavior as the Figure 1 central panel:
    # no forced equal aspect; axes fill the panel using the same Pe/Da limits.
    ax.set_xlabel("Pe", fontweight="bold", fontsize=FS_MAIN_LABEL)
    ax.set_ylabel("Da", fontweight="bold", fontsize=FS_MAIN_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=FS_MAIN_TICK, width=1.0, length=5)
    ax.grid(True, alpha=0.18, color="w", linewidth=0.6)

    cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.025, fraction=0.046)
    cbar.set_label(cbar_label, fontsize=FS_CBAR_LABEL, fontweight="bold")
    cbar.ax.tick_params(labelsize=FS_CBAR_TICK)
    cbar.ax.yaxis.set_major_formatter(COMPACT)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print(f"[Done] Saved figure: {out_png}")


# -----------------------------
# Main
# -----------------------------
def main():
    if not os.path.isfile(IN_H5):
        raise FileNotFoundError(f"Cannot find input H5: {IN_H5}")

    with h5py.File(IN_H5, "r") as f:
        Pe_list = require_dataset(f, "Pe_list").astype(np.float64)
        Da_list = require_dataset(f, "Da_list").astype(np.float64)

        mean_relL2 = require_dataset(f, "mean_relL2").astype(np.float64)
        std_relL2 = require_dataset(f, "std_relL2").astype(np.float64)
        mean_Seff_ratio = require_dataset(f, "mean_Seff_ratio").astype(np.float64)
        mean_iters = require_dataset(f, "mean_iters").astype(np.float64)

        Pe_star = float(f.attrs.get("Pe_star", 4.0))
        Da_star = float(f.attrs.get("Da_star", 2.0))

        Pe_min = float(f.attrs.get("Pe_min", float(Pe_list.min())))
        Pe_max = float(f.attrs.get("Pe_max", float(Pe_list.max())))
        Da_min = float(f.attrs.get("Da_min", float(Da_list.min())))
        Da_max = float(f.attrs.get("Da_max", float(Da_list.max())))

    nDa, nPe = mean_relL2.shape

    validate_grid_shape("std_relL2", std_relL2, nDa, nPe)
    validate_grid_shape("mean_Seff_ratio", mean_Seff_ratio, nDa, nPe)
    validate_grid_shape("mean_iters", mean_iters, nDa, nPe)

    if nPe != Pe_list.size or nDa != Da_list.size:
        raise ValueError(
            f"Shape mismatch: mean_relL2 {mean_relL2.shape}, "
            f"Pe_list size={Pe_list.size}, Da_list size={Da_list.size}. "
            "Expected mean_relL2 shape (nDa, nPe)."
        )

    print(f"[Info] Reading: {IN_H5}")
    print(f"[Info] Grid shape: nDa={nDa}, nPe={nPe}")
    print(f"[Info] Pe range: {Pe_list.min():g} to {Pe_list.max():g}")
    print(f"[Info] Da range: {Da_list.min():g} to {Da_list.max():g}")
    print(f"[Info] Central-panel ticks requested: {REQUESTED_TICKS.astype(int).tolist()}")

    # Figure 1: original summary figure, only central-panel x/y ticks changed.
    plot_summary_mean_relL2(
        Pe_list=Pe_list,
        Da_list=Da_list,
        mean_relL2=mean_relL2,
        std_relL2=std_relL2,
        Pe_star=Pe_star,
        Da_star=Da_star,
        Pe_min=Pe_min,
        Pe_max=Pe_max,
        Da_min=Da_min,
        Da_max=Da_max,
    )

    # Figure 2: same Pe-Da contour style/proportion as Figure 1 central panel.
    plot_standalone_contour(
        Pe_list=Pe_list,
        Da_list=Da_list,
        Z=mean_Seff_ratio,
        out_png=OUT_PNG_SEFF,
        title=r"Mean effective-source ratio",
        cbar_label=r"Mean $\|S_{\mathrm{eff}}\|_2 / \|S\|_2$",
        Pe_star=Pe_star,
        Da_star=Da_star,
        Pe_min=Pe_min,
        Pe_max=Pe_max,
        Da_min=Da_min,
        Da_max=Da_max,
    )

    # Figure 3: same Pe-Da contour style/proportion as Figure 1 central panel.
    plot_standalone_contour(
        Pe_list=Pe_list,
        Da_list=Da_list,
        Z=mean_iters,
        out_png=OUT_PNG_ITERS,
        title=r"Mean fixed-point iterations",
        cbar_label="Mean iterations",
        Pe_star=Pe_star,
        Da_star=Da_star,
        Pe_min=Pe_min,
        Pe_max=Pe_max,
        Da_min=Da_min,
        Da_max=Da_max,
    )


if __name__ == "__main__":
    main()
