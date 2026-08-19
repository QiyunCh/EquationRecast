#!/usr/bin/env python3
"""
Plot_Test3.py — Plot Test 3 results: 4-model comparison + Aitken/Anderson ablation.
"""
from __future__ import annotations
import os, h5py
import numpy as np
import matplotlib.pyplot as plt

# ---- Version-1 publication style: clear large bold fonts, dpi=300 ----
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 26,
    "axes.titleweight": "bold",
    "axes.labelsize": 23,
    "axes.labelweight": "bold",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 19,
    "lines.linewidth": 3.0,
    "lines.markersize": 9,
    "savefig.dpi": 300,
})

COL = {"canonical_dataonly": "tab:gray", "canonical_pinn": "tab:blue",
       "parametric": "tab:orange", "pino": "tab:green"}
LAB = {"canonical_dataonly": "Equation recast (data-only)",
       "canonical_pinn": "Equation recast (PINN-finetuned)",
       "parametric": "Parametric FNO", "pino": "PINO"}


def plot_compare():
    h5 = "results/test3_compare.h5"
    if not os.path.exists(h5):
        print(f"  skip: {h5} not found"); return

    with h5py.File(h5, "r") as f:
        Re = f["Re_list"][:]
        models = [k for k in f.keys() if k not in ("Re_list", "sources", "recast_iters",
                                                   "pino_ext")]
        err = {m: f[m]["err_l2"][:] for m in models}
        res = {m: f[m]["res_rel"][:] for m in models}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {"canonical_dataonly": "tab:gray", "canonical_pinn": "tab:blue",
              "parametric": "tab:orange", "pino": "tab:green"}
    labels = {"canonical_dataonly": "Recast (data-only canonical)",
              "canonical_pinn": "Recast (PINN-finetuned canonical)",
              "parametric": "Parametric FNO",
              "pino": "PINO (λ=0.5)"}
    for m in models:
        m_err = err[m].mean(axis=1)
        axes[0].plot(Re, m_err * 100, color=colors.get(m, "k"),
                     marker="o", markersize=4, label=labels.get(m, m))
        m_res = res[m].mean(axis=1)
        axes[1].plot(Re, m_res * 100, color=colors.get(m, "k"),
                     marker="o", markersize=4, label=labels.get(m, m))
    for ax, ylabel, title in zip(axes, ["rel L2 error (%)", "rel PDE residual (%)"],
                                  ["Error vs benchmark", "PDE residual ‖R‖/‖S‖"]):
        ax.axvspan(200, 300, alpha=0.1, color="gray", label="parametric/PINO train range")
        ax.axvline(250, color="red", linestyle="--", alpha=0.5, label="Re*=250")
        ax.set_xlabel("Re"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_yscale("log"); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    out = "results/Fig_Test3_compare.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def plot_aitken_anderson():
    h5 = "results/test3_aitken_vs_anderson.h5"
    if not os.path.exists(h5):
        print(f"  skip: {h5} not found"); return

    with h5py.File(h5, "r") as f:
        Re = f["Re_list"][:]
        schemes = [k for k in f.keys() if hasattr(f[k], "keys")]
        iters = {s: f[s]["iters"][:] for s in schemes}
        time = {s: f[s]["time"][:] for s in schemes}
        err = {s: f[s]["err_l2"][:] for s in schemes}
        nonconv = {s: f[s]["nonconv"][:] for s in schemes}

    # 3 panels stacked vertically; accuracy first (top), legend top-right.
    fig, axes = plt.subplots(3, 1, figsize=(9, 16), sharex=True)
    colors = {"aitken": "tab:blue", "anderson": "tab:orange", "underrelax": "tab:green"}
    labels = {"aitken": "Aitken $\\Delta^2$", "anderson": "Anderson ($m{=}3$)",
              "underrelax": "Under-relax ($\\omega{=}0.5$)"}
    for s in schemes:
        # Mask non-converged cells for iters/time (NaN); err stays valid everywhere
        it_mean = iters[s].astype(float).copy()
        it_mean[nonconv[s] > 0] = np.nan
        t_mean = time[s].copy(); t_mean[nonconv[s] > 0] = np.nan
        # No error bars (requested) — plot the mean trend only
        axes[0].plot(Re, err[s].mean(axis=1) * 100,
                     color=colors[s], marker="o", label=labels[s])          # accuracy
        axes[1].plot(Re, np.nanmean(it_mean, axis=1),
                     color=colors[s], marker="o", label=labels[s])          # iterations
        axes[2].plot(Re, np.nanmean(t_mean, axis=1) * 1000,
                     color=colors[s], marker="o", label=labels[s])          # wall time
    # Log y on all three (value ranges span 1-3 decades).
    for ax, ylabel, title in zip(axes,
                                 ["rel.\\ $L^2$ error (%)", "mean iterations",
                                  "mean wall time (ms)"],
                                 ["Final accuracy", "Convergence rate", "Inference cost"]):
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3)
    axes[2].set_xlabel("Re")
    axes[0].legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out = "results/Fig_Test3_aitken_anderson.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def plot_solver_time():
    h5 = "results/test3_solver_time.h5"
    if not os.path.exists(h5):
        print(f"  skip: {h5} not found"); return
    with h5py.File(h5, "r") as f:
        Re = f["Re_list"][:]
        cpu_s = f["solver_cpu_s"][:]
        gpu_s = f["solver_gpu_s"][:]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Re, cpu_s * 1000, marker="o", color="tab:orange", label="Solver CPU (NumPy)")
    ax.plot(Re, gpu_s * 1000, marker="o", color="tab:blue", label="Solver GPU (torch)")
    ax.set_xlabel("Re"); ax.set_ylabel("time per solve (ms)")
    ax.set_yscale("log"); ax.set_title("Numerical solver: CPU vs GPU")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    out = "results/Fig_Test3_solver_time.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def plot_combined_timing():
    """All-in-one inference time: recast / parametric / PINO / numerical solver."""
    h5_inf = "results/test3_inference_time.h5"
    h5_slv = "results/test3_solver_time.h5"
    if not os.path.exists(h5_inf) or not os.path.exists(h5_slv):
        print(f"  skip combined timing: missing {h5_inf} or {h5_slv}"); return

    with h5py.File(h5_inf, "r") as f:
        Re_inf = f["Re_list"][:]
        inf_models = [k for k in f.keys() if k not in ("Re_list", "pino_ext")]
        inf_time = {m: f[m]["time_s"][:] for m in inf_models}
        inf_iters = {m: f[m]["iters"][:] for m in inf_models}
    with h5py.File(h5_slv, "r") as f:
        Re_slv = f["Re_list"][:]
        cpu_s = f["solver_cpu_s"][:]
        gpu_s = f["solver_gpu_s"][:]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"canonical_dataonly": "tab:gray", "canonical_pinn": "tab:blue",
              "parametric": "tab:orange", "pino": "tab:green"}
    labels = {"canonical_dataonly": "Recast (data-only canonical, Aitken)",
              "canonical_pinn":     "Recast (PINN-finetuned canonical, Aitken)",
              "parametric":         "Parametric FNO (single forward)",
              "pino":               "PINO (single forward)"}

    for m in inf_models:
        med = np.median(inf_time[m], axis=1) * 1000  # ms
        lo  = np.percentile(inf_time[m], 25, axis=1) * 1000
        hi  = np.percentile(inf_time[m], 75, axis=1) * 1000
        ax.plot(Re_inf, med, marker="o", color=colors.get(m, "k"), label=labels.get(m, m))
        ax.fill_between(Re_inf, lo, hi, color=colors.get(m, "k"), alpha=0.2)

    ax.plot(Re_slv, cpu_s * 1000, marker="s", color="tab:red",
            linestyle="--", label="Numerical solver (NumPy CPU)")
    ax.plot(Re_slv, gpu_s * 1000, marker="s", color="tab:purple",
            linestyle="--", label="Numerical solver (torch GPU)")

    ax.set_xlabel("Re"); ax.set_ylabel("inference time per sample (ms)")
    ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Test 3 — inference cost: surrogates vs numerical solver")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = "results/Fig_Test3_combined_timing.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def plot_matched_combined_timing():
    """Inference cost at the accuracy equation recast achieves. For each
    (Re, source) the target accuracy T is the BEST relative L2 error the recast
    reaches (its tightest tolerance). The wall-time each method needs to reach
    that same T is then read from its time-vs-accuracy (tolerance) curve.
    Single-forward surrogates (parametric FNO, PINO) cannot be tuned to an
    arbitrary accuracy and are excluded; only equation recast vs the numerical
    solver (CPU and GPU) is a fair comparison."""
    h5 = "results/test3_matched_accuracy_time.h5"
    if not os.path.exists(h5):
        print(f"  skip matched timing: {h5} not found"); return

    def time_at_T(tts, errs, T):
        idx = np.where(np.asarray(errs) <= T * (1.0 + 1e-9))[0]
        return float(tts[idx[0]]) if len(idx) else float(tts[-1])

    with h5py.File(h5, "r") as f:
        Re = f["Re_list"][:]
        n_src = int(f.attrs["n_sources"])
        series = {k: [] for k in ("recast", "solver_cpu", "solver_gpu")}
        Tmed = []
        for R in Re:
            acc = {k: [] for k in series}; Ts = []
            for si in range(n_src):
                g = f[f"Re{int(R)}_src{si:02d}"]
                re_e = g["canonical_dataonly_err"][:]
                T = float(re_e.min())                       # recast best accuracy
                Ts.append(T)
                acc["recast"].append(time_at_T(g["canonical_dataonly_time_s"][:], re_e, T))
                acc["solver_cpu"].append(time_at_T(g["solver_cpu_time_s"][:],
                                                   g["solver_cpu_err"][:], T))
                acc["solver_gpu"].append(time_at_T(g["solver_gpu_time_s"][:],
                                                   g["solver_gpu_err"][:], T))
            for k in series:
                series[k].append(np.median(acc[k]) * 1000.0)  # ms
            Tmed.append(np.median(Ts) * 100.0)                # %

    style = {
        "recast":     ("Equation recast",              "#1f77b4", "s", "-"),
        "solver_cpu": ("Numerical solver (NumPy CPU)", "tab:red",    "o", "--"),
        "solver_gpu": ("Numerical solver (torch GPU)", "tab:purple", "o", "--"),
    }
    fig, ax = plt.subplots(figsize=(11, 7))
    for k in ("recast", "solver_cpu", "solver_gpu"):
        lab, c, mk, ls = style[k]
        ax.plot(Re, series[k], marker=mk, linestyle=ls, color=c, label=lab,
                linewidth=3.0, markersize=10)
    ax.axvline(250, color="red", linestyle="--", linewidth=2.0, alpha=0.85)
    ax.set_xlabel("Re")
    ax.set_ylabel("wall time to reach recast accuracy (ms)")
    ax.set_title("Inference cost at equation-recast accuracy", fontsize=24)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    out = "results/Fig_Test3_matched_combined_timing.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}  (target acc per Re %: " +
          ", ".join(f'{t:.1f}' for t in Tmed) + ")")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    plot_compare()
    plot_aitken_anderson()
    plot_solver_time()
    plot_combined_timing()
    plot_matched_combined_timing()
