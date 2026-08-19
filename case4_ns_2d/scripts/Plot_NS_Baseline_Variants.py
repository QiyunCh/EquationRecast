#!/usr/bin/env python3
"""
Plot_NS_Baseline_Variants.py — three full Fig-4 variants that differ ONLY in
the left diagnostic column of the detail rows:

  A: error spectrum        E_{pred-bench}(k), with benchmark E_omega overlay
  B: PDE-residual spectrum E_R(k)  (reference-free)
  C: per-shell relative error ||err(k)|| / ||bench(k)||  (k <= 21 only)

Top row (two error panels) and the four field columns are identical to the
production Fig 4. dpi=300.
"""
from __future__ import annotations
import os, h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Train_PINN_Canonical import make_kgrids

COMPARE_H5 = "results/test3_compare.h5"
FIELDS_H5  = "results/test3_fields.h5"

RE_DETAIL = [50, 150, 250, 350]
RE_MARK   = 250
K_HARD    = 21
SYMLOG_LINTHRESH = 1e-1
N = 128
EPS = 1e-30

C_BENCH  = "red"
C_RECAST = "#1f77b4"
C_PARAM  = "#ff7f0e"
C_PINO   = "#2ca02c"

CURVES = [
    ("canonical_dataonly", "Equation recast", C_RECAST, "s"),
    ("parametric",         "Parametric FNO",  C_PARAM,  "o"),
    ("pino",               "PINO",            C_PINO,   "^"),
]
FIELD_METHODS = [
    ("benchmark",          "Benchmark",  C_BENCH),
    ("canonical_dataonly", "Eq. recast", C_RECAST),
    ("parametric",         "Param. FNO", C_PARAM),
    ("pino",               "PINO",       C_PINO),
]

FIGWIDTH, FIGHEIGHT = 26, 28
L2_ROW_RATIO, DETAIL_ROW_RATIO = 0.7, 1.0
DETAIL_HSPACE = 0.32
DETAIL_WIDTH_RATIOS = [0.9, 0.9, 1.45, 1.45, 1.45, 1.45]
DETAIL_WSPACE = 0.10

# ---- spectral helpers ----
_k1 = np.fft.fftfreq(N) * N
_KX, _KY = np.meshgrid(_k1, _k1, indexing="ij")
RBIN = np.floor(np.sqrt(_KX ** 2 + _KY ** 2) + 0.5).astype(np.int32)
KMAX = int(RBIN.max())
KK = np.arange(KMAX + 1)

def shell_sum(w):
    return np.bincount(RBIN.ravel(), weights=w.ravel(), minlength=KMAX + 1)

def spec(a):
    """Shell-summed energy in physical amplitude^2 units (FFT normalized by N^2)."""
    a = a - a.mean()
    return shell_sum(np.abs(np.fft.fft2(a) / a.size) ** 2)

# residual (torch, matches Test_Compare convention)
_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_Kx, _Ky, _ = make_kgrids(N, 1.0, _dev)
_Kx = _Kx.double(); _Ky = _Ky.double()
_K2 = _Kx ** 2 + _Ky ** 2
_K2inv = torch.where(_K2 > 0, 1.0 / _K2, torch.zeros_like(_K2))

def residual_field(om_np, S_np, Re):
    o = torch.tensor(om_np, dtype=torch.float64, device=_dev)
    s = torch.tensor(S_np, dtype=torch.float64, device=_dev)
    Oh = torch.fft.rfft2(o, dim=(-2, -1), norm="backward")
    ph = Oh * _K2inv
    u = torch.fft.irfft2(1j * _Ky * ph, s=o.shape, dim=(-2, -1), norm="backward")
    v = torch.fft.irfft2(-1j * _Kx * ph, s=o.shape, dim=(-2, -1), norm="backward")
    ox = torch.fft.irfft2(1j * _Kx * Oh, s=o.shape, dim=(-2, -1), norm="backward")
    oy = torch.fft.irfft2(1j * _Ky * Oh, s=o.shape, dim=(-2, -1), norm="backward")
    lp = torch.fft.irfft2(-_K2 * Oh, s=o.shape, dim=(-2, -1), norm="backward")
    return (u * ox + v * oy - (1.0 / float(Re)) * lp - s).cpu().numpy()


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


# ---- the three left-column diagnostics ----
def diag_A(ax, g, Re_show, row):
    """Error spectra (primary axis) + benchmark energy as a filled envelope on
    a SECONDARY axis, so the signal level cannot be misread as an error curve."""
    bench = g["benchmark"][:]
    Eb = spec(bench)
    bpeak = Eb.max()

    # RIGHT axis: benchmark energy envelope, its own range (context only)
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(bpeak * 1e-9, bpeak * 30)
    ax2.fill_between(KK, bpeak * 1e-9, Eb + EPS, color="0.55", alpha=0.20, zorder=0)
    ax2.plot(KK, Eb + EPS, color="0.45", lw=1.6, zorder=1)
    ax2.set_ylabel(r"benchmark energy $E_\omega(k)$", fontsize=18,
                   fontweight="bold", color="0.35", labelpad=2)
    ax2.tick_params(axis="y", labelsize=13, colors="0.35", pad=1)

    # LEFT axis: error spectra, range fitted to the errors themselves
    handles = []
    Ees = []
    for mid, lab, col in FIELD_METHODS[1:]:
        Ee = spec(g[mid][:] - bench)
        Ees.append(Ee)
        h, = ax.semilogy(KK, Ee + EPS, color=col, lw=2.9, zorder=3,
                         label=f"{lab} error")
        handles.append(h)
    emax = max(E.max() for E in Ees)
    ax.set_yscale("log")
    ax.set_ylim(emax * 1e-9, emax * 30)
    ax.axvline(K_HARD, color="black", linestyle=":", lw=2.0, alpha=0.7)
    ax.set_xlim(0, 40)
    ax.set_ylabel("error spectrum $E_{\\mathrm{err}}(k)$", fontsize=22,
                  fontweight="bold")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    if row == 0:
        import matplotlib.patches as mpatches
        handles.append(mpatches.Patch(facecolor="0.75", alpha=0.5,
                                      label=r"benchmark $E_\omega(k)$ (right axis)"))
        ax.legend(handles=handles, loc="upper right", fontsize=14,
                  framealpha=0.92)


def diag_B(ax, g, Re_show, row):
    """PDE residual spectrum (reference-free)."""
    S = g["S"][:]
    peak = None
    for mid, lab, col in FIELD_METHODS[1:]:
        R = residual_field(g[mid][:], S, Re_show)
        Er = spec(R)
        if peak is None or Er.max() > peak:
            peak = Er.max()
        ax.semilogy(KK, Er + EPS, color=col, lw=2.8, label=lab)
    ax.axvspan(K_HARD, 40, color="gray", alpha=0.12)
    ax.axvline(K_HARD, color="black", linestyle=":", lw=2.0, alpha=0.7)
    ax.set_ylim(peak * 1e-9, peak * 30)
    ax.set_xlim(0, 40)
    ax.set_ylabel("PDE residual spectrum $E_{R}(k)$", fontsize=22, fontweight="bold")
    if row == 0:
        ax.legend(loc="lower left", fontsize=16, framealpha=0.9)


def diag_C(ax, g, Re_show, row):
    """Per-shell relative error, k in [1, 21] (beyond: no benchmark energy)."""
    bench = g["benchmark"][:]
    Fb = np.fft.fft2(bench - bench.mean())
    den = shell_sum(np.abs(Fb) ** 2)
    for mid, lab, col in FIELD_METHODS[1:]:
        pr = g[mid][:]
        Fr = np.fft.fft2(pr - pr.mean())
        num = shell_sum(np.abs(Fr - Fb) ** 2)
        rel = np.sqrt(num / (den + EPS))
        ax.semilogy(KK[1:K_HARD + 1], rel[1:K_HARD + 1] + EPS,
                    color=col, lw=2.8, marker="o", markersize=4, label=lab)
    ax.axhline(1.0, color="k", ls=":", lw=1.8, alpha=0.6)
    ax.set_xlim(1, K_HARD)
    ax.set_ylim(1e-3, 3e1)
    ax.set_ylabel("per-shell rel. error", fontsize=22, fontweight="bold")
    if row == 0:
        ax.legend(loc="lower right", fontsize=16, framealpha=0.9)


DIAGS = {"A_errspec": (diag_A, "(A) error spectrum"),
         "B_residspec": (diag_B, "(B) PDE-residual spectrum"),
         "C_shellrel": (diag_C, "(C) per-shell relative error")}


DIFF_METHODS = [
    ("canonical_dataonly", "Eq. recast $-$ Ref."),
    ("parametric",         "Param. FNO $-$ Ref."),
    ("pino",               "PINO $-$ Ref."),
]


def make_variant(tag, fields_mode="raw"):
    diag_fn, _ = DIAGS[tag]
    with h5py.File(COMPARE_H5, "r") as f:
        Re = f["Re_list"][:].astype(float)
        l2 = {k: f[k]["err_l2"][:].mean(axis=1) * 100 for k, *_ in CURVES}
        rr = {k: f[k]["res_rel"][:].mean(axis=1) * 100 for k, *_ in CURVES}

    fh = h5py.File(FIELDS_H5, "r")
    si = int(fh["src_indices"][:][0])

    n_det = len(RE_DETAIL)
    fig = plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    outer = gridspec.GridSpec(1 + n_det, 1, figure=fig,
                              height_ratios=[L2_ROW_RATIO] + [DETAIL_ROW_RATIO] * n_det,
                              hspace=DETAIL_HSPACE)

    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.20)
    plot_error_panel(fig.add_subplot(top[0]), Re, l2,
                     "Rel. $L^2$ error (%)", "Accuracy vs Re",
                     yscale="linear", ylim=(1, 25), xlim=(50, 400))
    plot_error_panel(fig.add_subplot(top[1]), Re, rr,
                     "Rel. PDE res. (%)", "PDE residual vs Re", yscale="log",
                     legend=False)

    # spectrum spans cols 0:2; col 2 is a thin spacer for the right-axis
    # ticks/label of the spectrum panel; fields start at col 3
    n_fields = 3 if fields_mode == "absdiff" else 4
    wr = DETAIL_WIDTH_RATIOS[:2] + [0.34] + [1.45] * n_fields
    n_cols = len(wr)
    for row, Re_show in enumerate(RE_DETAIL):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_cols, subplot_spec=outer[1 + row],
            width_ratios=wr, wspace=DETAIL_WSPACE)
        g = fh[f"Re{Re_show}_src{si:02d}"]

        ax_sp = fig.add_subplot(inner[0, 0:2])
        diag_fn(ax_sp, g, Re_show, row)
        ax_sp.set_xlabel("radial wavenumber $k$", fontsize=22, fontweight="bold")
        ax_sp.set_title(f"Re = {Re_show}", fontsize=26, fontweight="bold",
                        color=("red" if Re_show == RE_MARK else "black"))
        ax_sp.grid(True, which="both", alpha=0.25)
        ax_sp.tick_params(labelsize=17)

        ref = g["benchmark"][:]
        if fields_mode == "absdiff":
            # 3 columns of |pred - benchmark| (Figure_4_absdiff style):
            # sequential inferno, per-row shared 0->max scale, smooth
            diffs = [np.abs(g[mid][:] - ref) for mid, _lab in DIFF_METHODS]
            dmax = max(float(d.max()) for d in diffs)
            norm = mcolors.Normalize(vmin=0.0, vmax=dmax)
            im = None; ax_last = None
            for ci, ((mid, lab), d) in enumerate(zip(DIFF_METHODS, diffs)):
                ax = fig.add_subplot(inner[0, 3 + ci])
                im = ax.imshow(d, origin="lower", aspect="equal", cmap="inferno",
                               norm=norm, interpolation="bilinear")
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(lab, fontsize=24, fontweight="bold")
                ax_last = ax
            cax = make_axes_locatable(ax_last).append_axes("right", size="5%", pad=0.08)
            cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=15)
            cb.set_label(r"$|\Delta\omega|$", fontsize=16)
        else:
            vmin = float(np.nanmin(ref)); vmax = float(np.nanmax(ref))
            norm = mcolors.SymLogNorm(linthresh=SYMLOG_LINTHRESH, vmin=vmin, vmax=vmax)
            im = None; ax_last = None
            for ci, (mid, short, _) in enumerate(FIELD_METHODS):
                ax = fig.add_subplot(inner[0, 3 + ci])
                im = ax.imshow(g[mid][:], origin="lower", aspect="equal",
                               cmap="RdBu_r", norm=norm)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(short, fontsize=26, fontweight="bold")
                ax_last = ax
            cax = make_axes_locatable(ax_last).append_axes("right", size="5%", pad=0.08)
            cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=15)

    suffix = "_absdiff" if fields_mode == "absdiff" else ""
    out = f"results/Fig_Test3_NS_baseline_{tag}{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    fh.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    for tag in DIAGS:
        make_variant(tag)
