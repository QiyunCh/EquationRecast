#!/usr/bin/env python3
"""
Plot_NS_Baseline.py — combined NS baseline comparison (paper Fig. 4), styled
after D:\\...\\5_NS_2D\\20260407\\Plot_3.py.

Top row    : two error panels — Rel. L2 error vs Re | Rel. PDE residual vs Re
             (equation recast, parametric FNO, PINO; Re*=250 marked red).
Detail rows: per Re in {50,150,250,350}:
               - energy spectrum (Benchmark / Equation recast / Parametric / PINO)
               - vorticity fields (Benchmark / Equation recast / Parametric / PINO)

The equation-recast prediction is the standard K_hard=21 recast (consistent with
the top error curves and with the reference Plot_3.py). dpi=300.
"""
from __future__ import annotations
import os, h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

COMPARE_H5 = "results/test3_compare.h5"
FIELDS_H5  = "results/test3_fields.h5"   # benchmark, canonical_dataonly (recast), parametric, pino
OUT_PNG    = "results/Fig_Test3_NS_baseline.png"

RE_DETAIL = [50, 150, 250, 350]
RE_MARK   = 250
K_HARD    = 21
SYMLOG_LINTHRESH = 1e-1

# Version-1 palette
C_BENCH  = "red"
C_RECAST = "#1f77b4"
C_PARAM  = "#ff7f0e"
C_PINO   = "#2ca02c"

# top error-panel curves: (compare-h5 key, label, colour, marker)
CURVES = [
    ("canonical_dataonly", "Equation recast", C_RECAST, "s"),
    ("parametric",         "Parametric FNO",  C_PARAM,  "o"),
    ("pino",               "PINO",            C_PINO,   "^"),
]
# spectrum / field methods: (h5 key, full label, short field-title, colour)
FIELD_METHODS = [
    ("benchmark",          "Benchmark",       "Benchmark",  C_BENCH),
    ("canonical_dataonly", "Equation recast", "Eq. recast", C_RECAST),
    ("parametric",         "Parametric FNO",  "Param. FNO", C_PARAM),
    ("pino",               "PINO",            "PINO",       C_PINO),
]

# layout (4 fields -> 5 columns: spectrum spans 2, then 4 fields)
FIGWIDTH, FIGHEIGHT = 30, 28
L2_ROW_RATIO, DETAIL_ROW_RATIO = 0.7, 1.0
DETAIL_HSPACE = 0.32
DETAIL_WIDTH_RATIOS = [0.9, 0.9, 1.45, 1.45, 1.45, 1.45]  # spectrum spans first two
DETAIL_WSPACE = 0.10


def radial_bins(N):
    k1 = np.fft.fftfreq(N) * N           # integer mode indices (matches reference)
    kx, ky = np.meshgrid(k1, k1, indexing="ij")
    return np.floor(np.sqrt(kx ** 2 + ky ** 2) + 0.5).astype(np.int32)


def radial_power_spectrum(a, rbin):
    # Vorticity is zero-mean on the periodic domain; remove any spurious DC
    # offset so the k=0 shell is not contaminated by a constant component.
    a = a - a.mean()
    power = np.abs(np.fft.fft2(a)) ** 2
    rmax = int(rbin.max())
    bin_sum = np.bincount(rbin.ravel(), weights=power.ravel(), minlength=rmax + 1)
    return np.arange(rmax + 1), bin_sum


def plot_error_panel(ax, Re, series, ylabel, title, yscale="log",
                     ylim=None, xlim=None, legend=True):
    for key, lab, col, mk in CURVES:
        ax.plot(Re, series[key], marker=mk, color=col, label=lab,
                linewidth=3.0, markersize=9)
    ax.set_yscale(yscale)
    ax.set_xlim(*(xlim if xlim is not None else (Re.min(), Re.max())))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))
    if yscale == "linear" and ylim is not None:
        ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.set_xlabel("Re", fontsize=24, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=24, fontweight="bold")
    ax.set_title(title, fontsize=26, fontweight="bold")
    ax.grid(True, which="major", linestyle="-", alpha=0.4)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.tick_params(axis="both", labelsize=18)
    ax.axvspan(200, 300, color="gray", alpha=0.10)
    ax.axvline(RE_MARK, color="red", linestyle="--", linewidth=2.0, alpha=0.85)
    if legend:
        ax.legend(loc="best", fontsize=20, framealpha=0.95)
    ax.figure.canvas.draw()
    for tl in ax.get_xticklabels():
        if tl.get_text().replace("−", "-") == str(RE_MARK):
            tl.set_color("red"); tl.set_fontweight("bold"); tl.set_fontsize(19)


def main():
    for p in (COMPARE_H5, FIELDS_H5):
        if not os.path.exists(p):
            print(f"missing {p}"); return

    with h5py.File(COMPARE_H5, "r") as f:
        Re = f["Re_list"][:].astype(float)
        l2 = {k: f[k]["err_l2"][:].mean(axis=1) * 100 for k, *_ in CURVES}
        rr = {k: f[k]["res_rel"][:].mean(axis=1) * 100 for k, *_ in CURVES}

    fh = h5py.File(FIELDS_H5, "r")
    si = int(fh["src_indices"][:][0])
    N = fh[f"Re{RE_DETAIL[0]}_src{si:02d}"]["benchmark"].shape[-1]
    rbin = radial_bins(N)

    def fetch(nm, Re_):
        return fh[f"Re{Re_}_src{si:02d}"][nm][:]

    n_det = len(RE_DETAIL)
    fig = plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    outer = gridspec.GridSpec(1 + n_det, 1, figure=fig,
                              height_ratios=[L2_ROW_RATIO] + [DETAIL_ROW_RATIO] * n_det,
                              hspace=DETAIL_HSPACE)

    # ---- top: two error panels ----
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.20)
    # L2 panel: linear [1,25]% over Re in [50,400] (reference Plot_3 convention)
    plot_error_panel(fig.add_subplot(top[0]), Re, l2,
                     "Rel. $L^2$ error (%)", "Accuracy vs Re",
                     yscale="linear", ylim=(1, 25), xlim=(50, 400))
    # PDE residual panel: log (param residual spans >2 decades)
    plot_error_panel(fig.add_subplot(top[1]), Re, rr,
                     "Rel. PDE res. (%)", "PDE residual vs Re",
                     yscale="log", legend=False)

    # ---- detail rows ----
    for row, Re_show in enumerate(RE_DETAIL):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 6, subplot_spec=outer[1 + row],
            width_ratios=DETAIL_WIDTH_RATIOS, wspace=DETAIL_WSPACE)

        # spectrum (spans first two columns)
        ax_sp = fig.add_subplot(inner[0, 0:2])
        Ebench_max = 1.0
        for mid, lab, _short, col in FIELD_METHODS:
            k, E = radial_power_spectrum(fetch(mid, Re_show), rbin)
            E = E + 1e-30
            if mid == "benchmark":
                Ebench_max = float(E.max())
                ax_sp.plot(k, E, color=col, linestyle="-", linewidth=3.2, label=lab)
            elif mid == "canonical_dataonly":
                line, = ax_sp.plot(k, E, color=col, linestyle="--", linewidth=3.4, label=lab)
                line.set_dashes([10, 4])
            else:
                ax_sp.plot(k, E, color=col, linestyle="--", linewidth=2.2, alpha=0.85, label=lab)
        ax_sp.axvline(K_HARD, color="black", linestyle=":", linewidth=2.0, alpha=0.7)
        ax_sp.set_yscale("log")
        ax_sp.set_ylim(Ebench_max * 1e-7, Ebench_max * 3.0)   # focus on meaningful range
        ax_sp.set_xlim(0, 40)
        ax_sp.set_xlabel("radial wavenumber $k$", fontsize=22, fontweight="bold")
        ax_sp.set_ylabel("shell energy", fontsize=22, fontweight="bold")
        ax_sp.set_title(f"Energy spectrum (Re = {Re_show})", fontsize=26, fontweight="bold",
                        color=("red" if Re_show == RE_MARK else "black"))
        ax_sp.grid(True, which="both", alpha=0.25)
        ax_sp.tick_params(labelsize=17)
        if row == 0:
            ax_sp.legend(loc="upper right", fontsize=18, framealpha=0.9,
                         edgecolor="gray", fancybox=True)

        # fields
        ref = fetch("benchmark", Re_show)
        vmin = float(np.nanmin(ref)); vmax = float(np.nanmax(ref))
        norm = mcolors.SymLogNorm(linthresh=SYMLOG_LINTHRESH, vmin=vmin, vmax=vmax)
        im = None; ax_last = None
        for ci, (mid, lab, short, _) in enumerate(FIELD_METHODS):
            ax = fig.add_subplot(inner[0, 2 + ci])
            arr = fetch(mid, Re_show)
            im = ax.imshow(arr, origin="lower", aspect="equal", cmap="RdBu_r", norm=norm)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(short, fontsize=26, fontweight="bold")
            ax_last = ax
        cax = make_axes_locatable(ax_last).append_axes("right", size="5%", pad=0.08)
        cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=15)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    fh.close()
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
