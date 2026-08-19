#!/usr/bin/env python3
"""
TestRecast_Helmholtz_Compare2Models.py

Compare two FNO models on equation-recast 1D Helmholtz:
    u''(x) + k^2 u(x) = -S(x),  x in (0,L),  u(0)=u(L)=0

Model 1: trained on data_0.785.h5           (canonical k*=0.785)
Model 2: trained on data_merged_0.785_1.099.h5  (k*=0.785 + extra k=1.099 data)

For each k, 20 fixed GRF sources are used with both models.
Outputs:
  - HDF5 with all solutions and errors  (skips recomputation if found)
  - One figure: relative L2 and relative pointwise errors for both models,
    plus secondary y-axis showing mean iteration count.
"""

import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from Helmholtz import se_cholesky, sample_grf_from_cov, solve_fd_dirichlet_scipy_sparse
from FNO1D import FNO1d

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
SEED = 13
N_SAMPLES = 20

k_star = 0.785

# Model checkpoints
MODEL1_PATH = "best_fno1d_model1.pt"
MODEL2_PATH = "best_fno1d_model2.pt"

# Training data HDF5 (for u_min / u_max)
DATA_H5_MODEL1 = "data_0.785.h5"
DATA_H5_MODEL2 = "data_merged_0.785_1.099.h5"

# k sweep
k_list = [
    0.630, 0.635, 0.640, 0.650, 0.660, 0.670, 0.680, 0.690,
    0.700, 0.720, 0.750, 0.785,
    0.810, 0.840, 0.860, 0.880,
    0.900, 0.910, 0.920, 0.925, 0.930, 0.935, 0.940,
    0.950, 1.000, 1.100, 1.200,
]

# Aitken adaptive damping parameters
ALPHA_INIT = 0.5   # initial relaxation factor
ALPHA_MIN  = 0.1   # floor: don't waste iterations near resonance
ALPHA_MAX  = 1.0   # ceiling: full Newton-like step

# Resonance points  (k_n = n*pi/L)
RESONANCE_1 = 0.628   # n=2
RESONANCE_2 = 0.942   # n=3
RESONANCE_3 = 1.257   # n=4

# Grid / GRF (match Helmholtz.py)
L = 10.0
dx = 0.05
N = int(L / dx) + 1
x = np.linspace(0.0, L, N, endpoint=True)

grf_l = 1.0
grf_sigma = 1.0

# Iteration
TOL = 1e-4
MAX_ITERS = 200
EPS = 1e-12

# Outputs
OUT_H5 = "recast_helmholtz_compare_2models.h5"
OUT_FIG = "recast_helmholtz_compare_2models.png"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_u_range(h5_path: str) -> tuple[float, float]:
    """Read solution_min / solution_max from training HDF5."""
    with h5py.File(h5_path, "r") as f:
        u_min = float(f.attrs["solution_min"])
        u_max = float(f.attrs["solution_max"])
    return u_min, u_max


def load_model(path: str, device: torch.device) -> torch.nn.Module:
    model = FNO1d(modes=64, width=64, in_channels=2, out_channels=1).to(device)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def normalize_source_maxabs(S: np.ndarray, eps: float = 1e-12):
    c = float(np.max(np.abs(S)))
    if c < eps:
        c = 1.0
    return S / c, c


def denormalize_u(u_norm: np.ndarray, umin: float, umax: float) -> np.ndarray:
    return 0.5 * (u_norm + 1.0) * (umax - umin) + umin


def fno_solve_with_source_scaling(
    model, device, S_raw, xcoord_channel, umin, umax
):
    S_norm, c = normalize_source_maxabs(S_raw)
    inp = np.stack([S_norm, xcoord_channel], axis=0)[None, ...]
    xb = torch.tensor(inp, dtype=torch.float32, device=device)
    with torch.no_grad():
        yb = model(xb)
    u_norm = yb.detach().cpu().numpy()[0, 0, :]
    u_scaled = denormalize_u(u_norm, umin, umax)
    u = u_scaled * c
    return u.astype(np.float64)


def iter_recast_fno(
    model, device, S, k, xcoord_channel, umin, umax, tol, max_iters,
    alpha_init=ALPHA_INIT, alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX,
) -> tuple[np.ndarray, int, float, float]:
    """
    Fixed-point iteration with Aitken Δ² adaptive relaxation.

    The unrelaxed residual is  r^(n) = ũ^(n) − u^(n).
    Aitken update:
        α^(n) = −α^(n−1) * <r^(n−1), r^(n) − r^(n−1)> / ‖r^(n) − r^(n−1)‖²
    clamped to [alpha_min, alpha_max].

    Returns (u, n_iters, fp_error, final_alpha).
    """
    u_old = np.ones_like(S, dtype=np.float64)
    delta = k * k - k_star * k_star
    alpha = alpha_init
    r_prev = None  # residual from previous iteration

    for it in range(1, max_iters + 1):
        # FNO prediction on recast source
        S_recast = S + delta * u_old
        u_tilde = fno_solve_with_source_scaling(
            model, device, S_recast, xcoord_channel, umin, umax
        )

        # Unrelaxed residual
        r_cur = u_tilde - u_old

        # Aitken update (from iteration 2 onward)
        if r_prev is not None:
            dr = r_cur - r_prev
            dr_norm2 = float(np.dot(dr, dr))
            if dr_norm2 > EPS:
                alpha = -alpha * float(np.dot(r_prev, dr)) / dr_norm2
                alpha = np.clip(alpha, alpha_min, alpha_max)

        # Relaxed update
        u_new = u_old + alpha * r_cur
        err = float(np.linalg.norm(u_new - u_old)) / (float(np.linalg.norm(u_old)) + EPS)

        r_prev = r_cur.copy()
        u_old = u_new

        if err < tol:
            return u_old, it, err, alpha

    return u_old, max_iters, err, alpha


def run_one_model(model, device, umin, umax, xcoord, S_norm_all, label):
    """Run recast iteration for all k values with one model.
    Returns rel_l2_mean, rel_pt_mean, iters_mean, results_dict."""
    n_k = len(k_list)
    rel_l2_mean = np.zeros(n_k, dtype=np.float64)
    rel_pt_mean = np.zeros(n_k, dtype=np.float64)
    iters_mean  = np.zeros(n_k, dtype=np.float64)
    results = {}

    for ik, k in enumerate(k_list):
        k_key = f"{k:.3f}"
        print(f"  [{label}] k={k:.3f}")

        # Benchmark FD
        u_num = np.zeros((N_SAMPLES, N), dtype=np.float64)
        for i in range(N_SAMPLES):
            u_num[i] = solve_fd_dirichlet_scipy_sparse(x, S_norm_all[i], k)

        # FNO recast
        u_fno = np.zeros((N_SAMPLES, N), dtype=np.float64)
        iters = np.zeros(N_SAMPLES, dtype=np.int32)
        fp_err = np.zeros(N_SAMPLES, dtype=np.float64)
        final_alpha = np.zeros(N_SAMPLES, dtype=np.float64)

        for i in range(N_SAMPLES):
            u_pred, n_it, fe, fa = iter_recast_fno(
                model, device, S_norm_all[i], k,
                xcoord, umin, umax, TOL, MAX_ITERS,
            )
            u_fno[i] = u_pred
            iters[i] = n_it
            fp_err[i] = fe
            final_alpha[i] = fa

        # Relative errors
        rel_l2 = np.zeros(N_SAMPLES, dtype=np.float64)
        rel_pt = np.zeros(N_SAMPLES, dtype=np.float64)
        for i in range(N_SAMPLES):
            diff = u_fno[i] - u_num[i]
            norm_num = float(np.linalg.norm(u_num[i]))
            rel_l2[i] = float(np.linalg.norm(diff)) / (norm_num + EPS)
            mean_abs_num = float(np.mean(np.abs(u_num[i])))
            rel_pt[i] = float(np.mean(np.abs(diff))) / (mean_abs_num + EPS)

        rel_l2_mean[ik] = np.mean(rel_l2)
        rel_pt_mean[ik] = np.mean(rel_pt)
        iters_mean[ik]  = np.mean(iters)

        print(f"    rel L2 = {rel_l2_mean[ik]:.6e},  rel ptwise = {rel_pt_mean[ik]:.6e},  mean iters = {iters_mean[ik]:.1f},  mean final α = {np.mean(final_alpha):.3f}")

        results[k_key] = dict(
            u_num=u_num, u_fno=u_fno, rel_l2=rel_l2, rel_pt=rel_pt,
            iters=iters, fp_err=fp_err, final_alpha=final_alpha, k=k,
        )

    return rel_l2_mean, rel_pt_mean, iters_mean, results


# ─────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────
def load_results_from_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        k_arr = f["k_list"][:]
        m1_l2 = f["model1/rel_l2_mean"][:]
        m1_pt = f["model1/rel_pt_mean"][:]
        m1_it = f["model1/iters_mean"][:]
        m2_l2 = f["model2/rel_l2_mean"][:]
        m2_pt = f["model2/rel_pt_mean"][:]
        m2_it = f["model2/iters_mean"][:]
    return k_arr, m1_l2, m1_pt, m1_it, m2_l2, m2_pt, m2_it


# ─────────────────────────────────────────────
# Plotting — hybrid linear-log y-axis + iteration secondary axis
# ─────────────────────────────────────────────

# Transition threshold: linear below, log-compressed above
Y_LIN_THRESH = 0.5
LOG_REGION_FRAC = 0.20  # fraction of total height for log region
COMPRESS = 0.006  # default; recomputed in plot_comparison()


def _hybrid_forward(y):
    """Map data → display: linear for y <= thresh, heavily compressed log above."""
    y = np.asarray(y, dtype=np.float64)
    out = np.empty_like(y)
    mask = y <= Y_LIN_THRESH
    out[mask] = y[mask]
    out[~mask] = Y_LIN_THRESH + COMPRESS * np.log10(
        np.maximum(y[~mask], Y_LIN_THRESH) / Y_LIN_THRESH
    )
    return out


def _hybrid_inverse(y):
    """Map display → data (inverse of _hybrid_forward)."""
    y = np.asarray(y, dtype=np.float64)
    out = np.empty_like(y)
    mask = y <= Y_LIN_THRESH
    out[mask] = y[mask]
    out[~mask] = Y_LIN_THRESH * 10.0 ** ((y[~mask] - Y_LIN_THRESH) / COMPRESS)
    return out


def _closest_idx(arr, val, tol=0.01):
    idx = int(np.argmin(np.abs(arr - val)))
    return idx if abs(arr[idx] - val) < tol else None


def _split_at_resonance(k_arr, y_arr, k_break=RESONANCE_2):
    """Split arrays into two segments at the resonance point."""
    mask_left = k_arr <= k_break
    mask_right = k_arr > k_break
    return (k_arr[mask_left], y_arr[mask_left],
            k_arr[mask_right], y_arr[mask_right])

def plot_comparison(k_arr, m1_l2, m1_pt, m1_it, m2_l2, m2_pt, m2_it, out_fig):
    """
    Nature-style single-panel figure:
      Left y-axis  – hybrid linear/log for rel L2 error
      Right y-axis – linear for mean iteration count
    Lines broken at k = 0.942 (RESONANCE_2).

    Style updated only:
      - unified colors with previous figure
      - larger fonts
      - bold axis labels
      - diamond markers for Model 1/2 error curves
      - only Model 1 iteration curve kept, black dashed, legend 'NO. Iter.'
      - legend includes k* and k_new markers
    """
    # ── rcParams ──
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    # Unified color palette
    C_M1 = "#2563EB"   # Model 1
    C_M2 = "#059669"   # Model 2
    C_ITER = "black"   # iteration curve
    C_RES = "#DC2626"  # resonance / highlighted markers

    fig, ax = plt.subplots(figsize=(12, 5))

    LW_ERR = 1.5
    LW_ITER = 1.8
    MS_ERR = 5
    MS_ITER = 4
    FS_LABEL = 18
    FS_TICK = 14
    FS_LEGEND = 13
    FS_TITLE = 20
    FS_RES = 12

    # ── Helper: plot a line split at resonance ──
    def _plot_split(axis, k_arr, y_arr, color, ls, marker, mfc, label,
                    lw=LW_ERR, ms_=MS_ERR, mew=0.6, zorder=3):
        kL, yL, kR, yR = _split_at_resonance(k_arr, y_arr)
        axis.plot(
            kL, yL, ls, color=color, marker=marker, markersize=ms_,
            linewidth=lw, markerfacecolor=mfc, markeredgecolor=color,
            markeredgewidth=mew, label=label, zorder=zorder
        )
        axis.plot(
            kR, yR, ls, color=color, marker=marker, markersize=ms_,
            linewidth=lw, markerfacecolor=mfc, markeredgecolor=color,
            markeredgewidth=mew, zorder=zorder
        )

    # ── Rel-L2 error lines (left axis) ──
    _plot_split(ax, k_arr, m1_l2, C_M1, "-", "D", C_M1,
                "Model 1", lw=LW_ERR, ms_=MS_ERR)
    _plot_split(ax, k_arr, m2_l2, C_M2, "-", "D", C_M2,
                "Model 2", lw=LW_ERR, ms_=MS_ERR)

    # ── Highlight marker at k* = 0.785 (Model 1 only) ──
    idx_kstar = _closest_idx(k_arr, 0.785)
    if idx_kstar is not None:
        ax.scatter(
            k_arr[idx_kstar],
            m1_l2[idx_kstar],
            marker="*",
            s=350,                    # keep same as previous figure
            color=C_RES,
            edgecolor="k",
            linewidth=1.0,
            zorder=10,                # top layer
            label=r"$k^*$ (canonical)"
        )

    # ── Highlight at k ≈ 1.099 for Model 2 ──
    idx_k2 = _closest_idx(k_arr, 1.099, tol=0.01)
    if idx_k2 is None:
        idx_k2 = _closest_idx(k_arr, 1.1, tol=0.01)
    if idx_k2 is not None:
        ax.scatter(
            k_arr[idx_k2],
            m2_l2[idx_k2],
            marker="o",
            s=120,                    # keep same as previous figure
            color=C_RES,
            edgecolor="k",
            linewidth=1.0,
            zorder=10,                # top layer
            label=r"$k_{new}$"
        )

    # ── Resonance vertical lines ──
    for res_k in [RESONANCE_1, RESONANCE_2, RESONANCE_3]:
        ax.axvline(x=res_k, color=C_RES, ls="--", lw=1.2, alpha=0.5)

    # ── Apply hybrid scale to left axis ──
    global COMPRESS
    all_err = np.concatenate([m1_l2, m2_l2])
    err_max = float(np.max(all_err))
    y_top = max(err_max * 10, 1.0)
    n_decades = np.log10(y_top / Y_LIN_THRESH)
    COMPRESS = Y_LIN_THRESH * LOG_REGION_FRAC / ((1.0 - LOG_REGION_FRAC) * n_decades)

    ax.set_yscale("function", functions=(_hybrid_forward, _hybrid_inverse))
    ax.set_ylim(0, y_top)

    explicit_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1e5, 1e15, 1e20]
    explicit_ticks = [t for t in explicit_ticks if t <= y_top * 1.05]
    ax.set_yticks(explicit_ticks)

    def _fmt_ytick(val, pos):
        if val < 1.0:
            return f"{val:.1f}"
        else:
            exp = int(round(np.log10(val + 1e-15)))
            return f"$10^{{{exp}}}$"

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_ytick))
    ax.axhline(y=Y_LIN_THRESH, color="gray", ls="-", lw=0.5, alpha=0.3)

    # ── Secondary y-axis (right): mean iterations, Model 1 only ──
    ax_iter = ax.twinx()

    m1_it_clip = np.clip(m2_it, 0, MAX_ITERS)

    _plot_split(
        ax_iter, k_arr, m1_it_clip, C_ITER, "--", "^", "none",
        "NO. Iter.", lw=LW_ITER, ms_=MS_ITER, mew=0.8, zorder=2
    )

    ax_iter.set_ylim(0, 205)
    ax_iter.set_yticks([0, 50, 100, 150, 200])
    ax_iter.set_ylabel("NO. Iter.", fontsize=FS_LABEL, fontweight="bold", color=C_ITER)
    ax_iter.tick_params(axis="y", labelsize=FS_TICK, colors=C_ITER)
    ax_iter.spines["right"].set_color(C_ITER)
    ax_iter.spines["right"].set_linewidth(1.0)

    # ── x-axis ──
    ax.set_xlim(0.62, 1.28)
    ax.set_xlabel("k", fontsize=FS_LABEL, fontweight="bold")

    normal_xticks = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
    ax.set_xticks(normal_xticks)
    ax.set_xticklabels([f"{v:.2f}" for v in normal_xticks], fontsize=FS_TICK)
    ax.tick_params(labelsize=FS_TICK)

    # Resonance labels on top x-axis
    ax2 = ax.secondary_xaxis("top")
    ax2.set_xticks([RESONANCE_1, RESONANCE_2, RESONANCE_3])
    ax2.set_xticklabels(
        [f"$k_2$={RESONANCE_1}", f"$k_3$={RESONANCE_2}", f"$k_4$={RESONANCE_3}"],
        fontsize=FS_RES, color=C_RES,
    )
    ax2.tick_params(axis="x", colors=C_RES, direction="in", length=5, width=1.0)

    # ── Labels / legend / grid ──
    ax.set_ylabel("Mean Relative Error", fontsize=FS_LABEL, fontweight="bold")
    ax.set_title(
        r"Helmholtz Equation Recast Test ($u_{xx}+k^2u=-S$)",
        fontsize=FS_TITLE, fontweight="bold", pad=18,
    )

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_iter.get_legend_handles_labels()
    leg = ax.legend(
        lines1 + lines2, labels1 + labels2,
        fontsize=FS_LEGEND, loc="center right",
        frameon=True, framealpha=0.92, edgecolor="#cccccc",
        fancybox=False, borderpad=0.8, handlelength=2.5,
    )
    leg.get_frame().set_linewidth(0.8)

    ax.grid(True, which="major", alpha=0.15, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved figure: {out_fig}")
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    k_arr = np.array(k_list, dtype=np.float64)

    # ── Check for existing results ──
    if os.path.isfile(OUT_H5):
        print(f"[Info] Found existing results: {OUT_H5}. Loading directly.")
        k_arr, m1_l2, m1_pt, m1_it, m2_l2, m2_pt, m2_it = load_results_from_h5(OUT_H5)
        plot_comparison(k_arr, m1_l2, m1_pt, m1_it, m2_l2, m2_pt, m2_it, OUT_FIG)
        return

    # ── Compute from scratch ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    # Load u ranges from training data
    u_min1, u_max1 = load_u_range(DATA_H5_MODEL1)
    u_min2, u_max2 = load_u_range(DATA_H5_MODEL2)
    print(f"[Info] Model 1 u_range: [{u_min1:.6f}, {u_max1:.6f}]")
    print(f"[Info] Model 2 u_range: [{u_min2:.6f}, {u_max2:.6f}]")

    # Load models
    model1 = load_model(MODEL1_PATH, device)
    model2 = load_model(MODEL2_PATH, device)
    print(f"[Info] Loaded {MODEL1_PATH} and {MODEL2_PATH}")

    # x-coordinate channel [-1, 1]
    xcoord = np.linspace(-1.0, 1.0, N, endpoint=True).astype(np.float64)

    # Generate 20 GRF sources (normalized)
    Lc = se_cholesky(x, l=grf_l, sigma=grf_sigma)
    rng = np.random.default_rng(SEED)

    S_norm_all = np.zeros((N_SAMPLES, N), dtype=np.float64)
    S_factor_all = np.zeros(N_SAMPLES, dtype=np.float64)
    S_raw_all = np.zeros((N_SAMPLES, N), dtype=np.float64)

    for i in range(N_SAMPLES):
        S_raw = sample_grf_from_cov(Lc, rng)
        S_norm, c = normalize_source_maxabs(S_raw)
        S_raw_all[i] = S_raw
        S_norm_all[i] = S_norm
        S_factor_all[i] = c

    # Run both models
    print("\n===== Model 1 (k*=0.785 only) =====")
    m1_l2, m1_pt, m1_it, res1 = run_one_model(
        model1, device, u_min1, u_max1, xcoord, S_norm_all, "M1"
    )

    print("\n===== Model 2 (k*=0.785 + k=1.099) =====")
    m2_l2, m2_pt, m2_it, res2 = run_one_model(
        model2, device, u_min2, u_max2, xcoord, S_norm_all, "M2"
    )

    # ── Save HDF5 ──
    with h5py.File(OUT_H5, "w") as f:
        f.attrs["seed"] = SEED
        f.attrs["N_samples"] = N_SAMPLES
        f.attrs["k_star"] = k_star
        f.attrs["L"] = L
        f.attrs["dx"] = dx
        f.attrs["N"] = N
        f.attrs["tol"] = TOL
        f.attrs["max_iters"] = MAX_ITERS

        f.create_dataset("k_list", data=k_arr)
        f.attrs["alpha_init"] = ALPHA_INIT
        f.attrs["alpha_min"] = ALPHA_MIN
        f.attrs["alpha_max"] = ALPHA_MAX
        f.create_dataset("x_phys", data=x)
        f.create_dataset("sources_raw", data=S_raw_all)
        f.create_dataset("sources_norm", data=S_norm_all)
        f.create_dataset("sources_factor", data=S_factor_all)

        for tag, res, rl2, rpt, rit, um, ux in [
            ("model1", res1, m1_l2, m1_pt, m1_it, u_min1, u_max1),
            ("model2", res2, m2_l2, m2_pt, m2_it, u_min2, u_max2),
        ]:
            gm = f.create_group(tag)
            gm.attrs["u_min"] = um
            gm.attrs["u_max"] = ux
            gm.create_dataset("rel_l2_mean", data=rl2)
            gm.create_dataset("rel_pt_mean", data=rpt)
            gm.create_dataset("iters_mean", data=rit)

            for k_key, rd in res.items():
                gb = gm.create_group(k_key)
                gb.attrs["k"] = rd["k"]
                gb.create_dataset("numerical", data=rd["u_num"])
                gb.create_dataset("FNO", data=rd["u_fno"])
                gb.create_dataset("rel_l2", data=rd["rel_l2"])
                gb.create_dataset("rel_pt", data=rd["rel_pt"])
                gb.create_dataset("iters", data=rd["iters"])
                gb.create_dataset("fp_err", data=rd["fp_err"])
                gb.create_dataset("final_alpha", data=rd["final_alpha"])

    print(f"\n[Done] Saved HDF5: {OUT_H5}")

    # ── Plot ──
    plot_comparison(k_arr, m1_l2, m1_pt, m1_it, m2_l2, m2_pt, m2_it, OUT_FIG)


if __name__ == "__main__":
    main()