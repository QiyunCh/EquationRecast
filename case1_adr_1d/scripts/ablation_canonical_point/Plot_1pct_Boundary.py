#!/usr/bin/env python3
"""
Plot_1pct_Boundary.py — Overlay 1% relative-L2 error boundaries.

Reads scan H5 files from results/, draws the 1% error contour for each canonical
and stage on a single (Pe, Da) plane. Also generates per-canonical heatmaps for
sanity-checking.
"""
from __future__ import annotations
import os, glob, h5py
import numpy as np
import matplotlib.pyplot as plt

# ---- Version-1 publication style: clear large bold fonts, dpi=300 ----
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 26,
    "axes.titleweight": "bold",
    "axes.labelsize": 24,
    "axes.labelweight": "bold",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "lines.linewidth": 2.6,
    "savefig.dpi": 300,
})

RESULTS_DIR = "results"
ERR_LEVEL = 0.01  # 1% relative L2

# 2x3 publication heatmap layout (stage 1): row-major order
HEATMAP_GRID = [(1.0, 1.0), (2.0, 2.0), (2.0, 4.0),
                (2.0, 6.0), (10.0, 10.0), (25.0, 2.0)]

CANON_COLORS = {
    (2.0, 4.0):   "tab:blue",
    (10.0, 10.0): "tab:orange",
    (2.0, 25.0):  "tab:green",
    (25.0, 2.0):  "tab:red",
    (1.0, 1.0):   "tab:purple",
    (2.0, 6.0):   "tab:brown",
    (2.0, 2.0):   "tab:pink",
}


def load_scan(h5_path):
    with h5py.File(h5_path, "r") as f:
        return {
            "Pe": f["Pe_list"][:],
            "Da": f["Da_list"][:],
            "err": f["rel_l2"][:],
            "iters": f["iters_mean"][:],
            "nonconv": f["nonconv"][:],
            "Pe_star": float(f.attrs["Pe_star"]),
            "Da_star": float(f.attrs["Da_star"]),
        }


def plot_overlay(stage: str, out_png: str):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"scan_Pe*_Da*_{stage}.h5")))
    if not files:
        print(f"  no scan files found for stage={stage}")
        return

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    for h5 in files:
        d = load_scan(h5)
        Pe, Da = d["Pe"], d["Da"]
        # Mask only cells where all 20 sources failed (full nonconvergence).
        # Cells with partial nonconvergence still report mean error over converged sources.
        err = d["err"].copy()
        err[d["nonconv"] >= 20] = np.nan
        PE, DA = np.meshgrid(Pe, Da)

        key = (d["Pe_star"], d["Da_star"])
        color = CANON_COLORS.get(key, "k")
        label = f"({d['Pe_star']:g}, {d['Da_star']:g})"

        # 1% contour
        with np.errstate(invalid="ignore"):
            cs = ax.contour(PE, DA, err, levels=[ERR_LEVEL], colors=[color], linewidths=2.0)
        # Mark the canonical point
        ax.plot(d["Pe_star"], d["Da_star"], marker="*", markersize=18,
                color=color, markeredgecolor="black", linewidth=0.5)
        # Dummy line for legend
        ax.plot([], [], color=color, lw=2.0, label=f"canonical {label}")

    ax.set_xlim(Pe.min(), Pe.max())
    ax.set_ylim(Da.min(), Da.max())
    ax.set_xlabel("Pe")
    ax.set_ylabel("Da")
    ax.set_title(f"1% relative-L2 error boundary per canonical ({stage})")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png}")


def plot_per_canonical_heatmaps(stage: str, out_png: str):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"scan_Pe*_Da*_{stage}.h5")))
    if not files:
        return

    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5), squeeze=False)
    for ax, h5 in zip(axes[0], files):
        d = load_scan(h5)
        Pe, Da = d["Pe"], d["Da"]
        err = d["err"].copy()
        err[d["nonconv"] >= 20] = np.nan
        im = ax.imshow(np.log10(err.clip(1e-6, 1e2)),
                       origin="lower",
                       extent=[Pe.min(), Pe.max(), Da.min(), Da.max()],
                       aspect="auto", cmap="viridis", vmin=-4, vmax=0)
        ax.contour(np.linspace(Pe.min(), Pe.max(), err.shape[1]),
                   np.linspace(Da.min(), Da.max(), err.shape[0]),
                   err, levels=[ERR_LEVEL], colors="red", linewidths=2.0)
        ax.plot(d["Pe_star"], d["Da_star"], marker="*", markersize=18,
                color="red", markeredgecolor="black")
        ax.set_xlabel("Pe"); ax.set_ylabel("Da")
        ax.set_title(f"canonical=({d['Pe_star']:g}, {d['Da_star']:g}) [{stage}]")
        plt.colorbar(im, ax=ax, label="log10 rel-L2")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png}")


def plot_heatmaps_grid_stage1(out_png: str):
    """Publication 2x3 heatmap grid for the stage-1 (data-only) canonicals.
    Row 1: (1,1),(2,2),(2,4);  Row 2: (2,6),(10,10),(25,2).
    Single shared colorbar on the far right, no titles, canonical (Pe,Da)
    labelled in red next to the star."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5))
    im = None
    for idx, (pe, da) in enumerate(HEATMAP_GRID):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        h5 = os.path.join(RESULTS_DIR, f"scan_Pe{pe:g}_Da{da:g}_stage1.h5")
        if not os.path.exists(h5):
            ax.set_visible(False); continue
        d = load_scan(h5)
        Pe, Da = d["Pe"], d["Da"]
        err = d["err"].copy()
        err[d["nonconv"] >= 20] = np.nan
        im = ax.imshow(np.log10(err.clip(1e-6, 1e2)), origin="lower",
                       extent=[Pe.min(), Pe.max(), Da.min(), Da.max()],
                       aspect="auto", cmap="viridis", vmin=-4, vmax=0)
        ax.contour(np.linspace(Pe.min(), Pe.max(), err.shape[1]),
                   np.linspace(Da.min(), Da.max(), err.shape[0]),
                   err, levels=[ERR_LEVEL], colors="red", linewidths=2.4)
        ax.plot(d["Pe_star"], d["Da_star"], marker="*", markersize=26,
                color="red", markeredgecolor="black", markeredgewidth=1.2)
        # red coordinate label next to the star (offset, clamped inside frame)
        dx = -7.5 if d["Pe_star"] > 40 else 2.5
        dy = -4.0 if d["Da_star"] > 44 else 2.5
        ax.text(d["Pe_star"] + dx, d["Da_star"] + dy,
                f"({d['Pe_star']:g}, {d['Da_star']:g})",
                color="red", fontsize=19, fontweight="bold",
                ha="left", va="bottom")
        # reduce labels: y only on left column, x only on bottom row
        if c == 0:
            ax.set_ylabel("Da")
        if r == 1:
            ax.set_xlabel("Pe")

    # shared colorbar on the far right
    fig.subplots_adjust(left=0.06, right=0.90, top=0.98, bottom=0.07,
                        wspace=0.16, hspace=0.14)
    cax = fig.add_axes([0.915, 0.07, 0.018, 0.91])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10}$ rel.\ $L^2$ error")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png}")


def print_summary_table(n_sources: int = 20):
    """Use the relaxed mask: a cell is 'usable' if at least one of n_sources converged."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "scan_*.h5")))
    print("\n========== Test 1 summary table ==========")
    print(f"{'file':<35} {'Pe*':>5} {'Da*':>5}  {'med_err':>10}  {'%err<1%':>9}  {'%full_nc':>9}  {'med_iter':>9}")
    for h5 in files:
        d = load_scan(h5)
        err = d["err"].copy()
        nc = d["nonconv"].copy()
        valid = (nc < n_sources) & np.isfinite(err)  # partial nonconv still valid
        med_err = float(np.nanmedian(err[valid])) if valid.any() else float("nan")
        pct1 = 100.0 * ((err < ERR_LEVEL) & valid).sum() / err.size
        pct_full_nc = 100.0 * (nc >= n_sources).sum() / nc.size
        med_iter = float(np.nanmedian(d["iters"]))
        name = os.path.basename(h5)
        print(f"{name:<35} {d['Pe_star']:>5.1f} {d['Da_star']:>5.1f}  {med_err:>10.3e}  {pct1:>8.1f}%  {pct_full_nc:>8.1f}%  {med_iter:>9.1f}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for stage in ("stage1", "stage2"):
        plot_overlay(stage, os.path.join(RESULTS_DIR, f"Fig_1pct_boundary_{stage}.png"))
    # stage 1 heatmaps use the publication 2x3 grid; stage 2 keeps the generic strip
    plot_heatmaps_grid_stage1(os.path.join(RESULTS_DIR, "Fig_heatmaps_stage1.png"))
    plot_per_canonical_heatmaps("stage2", os.path.join(RESULTS_DIR, "Fig_heatmaps_stage2.png"))
    # Compare stage1 vs stage2 boundaries side by side
    files_s1 = sorted(glob.glob(os.path.join(RESULTS_DIR, "scan_Pe*_Da*_stage1.h5")))
    files_s2 = sorted(glob.glob(os.path.join(RESULTS_DIR, "scan_Pe*_Da*_stage2.h5")))
    if files_s1 and files_s2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        for stage, ax in zip(("stage1", "stage2"), axes):
            files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"scan_Pe*_Da*_{stage}.h5")))
            for h5 in files:
                d = load_scan(h5)
                err = d["err"].copy()
                err[d["nonconv"] >= 20] = np.nan
                PE, DA = np.meshgrid(d["Pe"], d["Da"])
                color = CANON_COLORS.get((d["Pe_star"], d["Da_star"]), "k")
                with np.errstate(invalid="ignore"):
                    ax.contour(PE, DA, err, levels=[ERR_LEVEL], colors=[color], linewidths=2.0)
                ax.plot(d["Pe_star"], d["Da_star"], marker="*", markersize=14,
                        color=color, markeredgecolor="black")
                ax.plot([], [], color=color, lw=2.0,
                        label=f"({d['Pe_star']:g},{d['Da_star']:g})")
            ax.set_xlabel("Pe"); ax.set_ylabel("Da")
            ax.set_title(f"1% boundary ({stage})")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(RESULTS_DIR, "Fig_1pct_boundary_compare.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")

    print_summary_table()
