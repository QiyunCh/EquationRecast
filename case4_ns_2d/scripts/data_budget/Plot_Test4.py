#!/usr/bin/env python3
"""
Plot_Test4.py — Visualize Test 4 results (budget vs error/residual).
"""
from __future__ import annotations
import os, h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---- Version-1 publication style: clear, large bold fonts ----
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 26,
    "axes.labelsize": 23,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 19,
    "lines.linewidth": 3.0,
    "lines.markersize": 11,
    "savefig.dpi": 300,
})

# Pretty labels for the legend (Version-1 phrasing)
PRETTY = {
    "canonical_data":     "Equation recast",
    "parametric":         "Parametric FNO",
    "pino":               "PINO",
    "redistributed_data": "Redistributed recast",
}


def parse_name(name):
    """Extract (model_type, budget) from checkpoint basename."""
    if "canonical_data_N" in name:
        return ("canonical_data", int(name.split("N")[-1]))
    if "canonical_pinn_N" in name:
        return ("canonical_pinn", int(name.split("N")[-1]))
    if "parametric_N" in name:
        return ("parametric", int(name.split("N")[-1]))
    if "pino_ext_N" in name:
        return ("pino_ext", int(name.split("N")[-1]))
    if "pino_N" in name:
        return ("pino", int(name.split("N")[-1]))
    if "redistributed_data_N" in name:
        return ("redistributed_data", int(name.split("N")[-1]))
    if "redistributed_data" in name:
        return ("redistributed_data", 1500)
    if "redistributed_pinn" in name:
        return ("redistributed_pinn", 1500)
    return (name, 0)


def main():
    h5 = "results/test4_compare.h5"
    if not os.path.exists(h5):
        print(f"missing {h5}"); return
    with h5py.File(h5, "r") as f:
        Re = f["Re_list"][:]
        models = [k for k in f.keys() if k not in ("Re_list", "sources")]
        err = {m: f[m]["err_l2"][:] for m in models}
        res = {m: f[m]["res_rel"][:] for m in models}

    # Group by (model_type, budget)
    grouped = {}
    for m in models:
        mt, B = parse_name(m)
        grouped[(mt, B)] = m

    budgets = sorted({B for (_, B) in grouped if B > 0 and B != 1500 or (B == 1500 and any(t.startswith("canonical") or t.startswith("parametric") or t.startswith("pino") for (t, b) in grouped if b == B))})
    budgets = sorted({B for (_, B) in grouped if B > 0})
    # canonical_pinn excluded from Test 4 reporting (per request)
    types_order = ["canonical_data", "parametric", "pino", "redistributed_data"]
    colors = {"canonical_data": "tab:gray",
              "parametric": "tab:orange", "pino": "tab:green",
              "redistributed_data": "tab:purple"}

    # Plot 1 (s10a): error vs Re at each budget, 2x2 grid
    nB = len(budgets)
    ncol = 2
    nrow = int(np.ceil(nB / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 6.0 * nrow),
                             sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    for j, B in enumerate(budgets):
        ax = axes[j]
        for mt in types_order:
            if (mt, B) not in grouped:
                continue
            m = grouped[(mt, B)]
            e = err[m].mean(axis=1) * 100
            ax.plot(Re, e, marker="o", color=colors[mt], label=PRETTY.get(mt, mt))
        ax.set_title(f"N = {B}", fontsize=24, fontweight="bold")
        ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3)
        ax.axvline(250, color="red", linestyle="--", linewidth=2.0, alpha=0.7)
        if j % ncol == 0:
            ax.set_ylabel("Rel. $L^2$ error (%)", fontweight="bold")
        if j // ncol == nrow - 1:
            ax.set_xlabel("Re", fontweight="bold")
        if j == 0:
            ax.legend(loc="lower right", framealpha=0.95, fontsize=16)
    for k in range(nB, len(axes)):
        axes[k].set_visible(False)
    fig.tight_layout()
    out = "results/Fig_Test4_error_vs_Re.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")

    # Plot 2 (s10b): budget scaling, vertical stack (L2 top, residual bottom),
    # linear y, Fig.4-style labels, single legend top-right.
    fig, axes = plt.subplots(2, 1, figsize=(9, 12), sharex=True)
    for mt in types_order:
        bs = [B for B in budgets if (mt, B) in grouped]
        if not bs:
            continue
        e_means = [err[grouped[(mt, B)]].mean() * 100 for B in bs]
        r_means = [res[grouped[(mt, B)]].mean() * 100 for B in bs]
        axes[0].plot(bs, e_means, marker="o", color=colors[mt], label=PRETTY.get(mt, mt))
        axes[1].plot(bs, r_means, marker="o", color=colors[mt], label=PRETTY.get(mt, mt))
    for ax, ylabel, title in zip(
            axes, ["Rel. $L^2$ error (%)", "Rel. PDE res. (%)"],
            ["Accuracy vs data budget", "PDE residual vs data budget"]):
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontsize=24, fontweight="bold")
        ax.set_xscale("log"); ax.set_yscale("linear")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks(budgets)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(axis="both", labelsize=18)
    axes[1].set_xlabel("data budget $N$", fontweight="bold")
    axes[0].legend(loc="upper right", framealpha=0.95)   # single legend
    fig.tight_layout()
    out = "results/Fig_Test4_budget_vs_err.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    main()
