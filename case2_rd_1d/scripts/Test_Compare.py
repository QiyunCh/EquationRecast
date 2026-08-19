"""
TestRecast_ReactionDiffusion_Compare2Models.py

Compare two FNO models on equation-recast 1D reaction–diffusion:
    -u''(x) + beta*u(x) = S(x),  x in (0,L),  u(0)=u(L)=0

Model 1: trained on data_beta_0.785.h5  (canonical beta*=0.785)
Model 2: trained on data_merged_beta_0.785_0.050.h5  (same beta* + extra beta=0.05 data)

For each beta, 20 fixed GRF sources are used with both models.
Outputs:
  - HDF5 with all solutions and errors
  - One figure: relative L2 and relative pointwise errors for both models
"""

import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from ReactionDiffusion import se_cholesky, sample_grf_from_cov, solve_fd_dirichlet_scipy_sparse
from FNO1D import FNO1d

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
SEED = 13
N_SAMPLES = 20

beta_star = 0.785

# Model checkpoints
MODEL1_PATH = "best_fno1d_model1.pt"
MODEL2_PATH = "best_fno1d_model2.pt"

# Training data HDF5 (for u_min / u_max)
DATA_H5_MODEL1 = "data_beta_0.785.h5"
DATA_H5_MODEL2 = "data_merged_beta_0.785_0.050.h5"

# Beta sweep
k_list = [
    0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.09, 0.10,
    0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.60, 0.70, 0.785, 0.80, 0.90, 1.00,
    2, 4, 6, 8, 10,
]

# Damping: 0.1 for beta < 0.5 or beta > 1.0; 0.8 otherwise
damping_list = [0.1 if (b < 0.5 or b > 1.0) else 0.8 for b in k_list]

# Grid / GRF
L = 10.0
dx = 0.05
N = int(L / dx) + 1
x = np.linspace(0.0, L, N, endpoint=True)

grf_l = 0.5
grf_sigma = 1.0

# Iteration
TOL = 1e-4
MAX_ITERS = 500
EPS = 1e-12

# Outputs
OUT_H5 = "recast_rd_compare_2models.h5"
OUT_FIG = "recast_rd_compare_2models.png"


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
    model: torch.nn.Module,
    device: torch.device,
    S_raw: np.ndarray,
    xcoord_channel: np.ndarray,
    umin: float,
    umax: float,
) -> np.ndarray:
    S_norm, c = normalize_source_maxabs(S_raw)
    inp = np.stack([S_norm, xcoord_channel], axis=0)[None, ...]
    xb = torch.tensor(inp, dtype=torch.float32, device=device)
    with torch.no_grad():
        yb = model(xb)
    u_norm = yb.detach().cpu().numpy()[0, 0, :]
    u_scaled = denormalize_u(u_norm, umin, umax)
    u = u_scaled * c
    u[0] = 0.0
    u[-1] = 0.0
    return u.astype(np.float64)


def iter_recast_fno(
    model, device, S, beta, alpha, xcoord_channel, umin, umax, tol, max_iters
) -> tuple[np.ndarray, int, float]:
    u_old = np.zeros_like(S, dtype=np.float64)
    delta = beta - beta_star

    for it in range(1, max_iters + 1):
        S_eff = S - delta * u_old
        u_tilde = fno_solve_with_source_scaling(model, device, S_eff, xcoord_channel, umin, umax)
        u_new = (1.0 - alpha) * u_old + alpha * u_tilde
        u_new[0] = 0.0
        u_new[-1] = 0.0

        err = float(np.linalg.norm(u_new - u_old)) / (float(np.linalg.norm(u_old)) + EPS)
        u_old = u_new
        if err < tol:
            return u_old, it, err

    return u_old, max_iters, err


def run_one_model(
    model, device, umin, umax, xcoord, S_norm_all, label
):
    """Run recast iteration for all betas with one model. Returns dict of results."""
    n_betas = len(k_list)
    rel_l2_mean = np.zeros(n_betas, dtype=np.float64)
    rel_pt_mean = np.zeros(n_betas, dtype=np.float64)

    # Store per-beta solutions for HDF5
    results = {}

    for ib, (beta, alpha) in enumerate(zip(k_list, damping_list)):
        beta_key = f"{beta:.3f}"
        print(f"  [{label}] beta={beta:.3f}, alpha={alpha:.2f}")

        # --- Benchmark (FD) ---
        u_num = np.zeros((N_SAMPLES, N), dtype=np.float64)
        for i in range(N_SAMPLES):
            u_num[i] = solve_fd_dirichlet_scipy_sparse(x, S_norm_all[i], beta)

        # --- FNO recast ---
        u_fno = np.zeros((N_SAMPLES, N), dtype=np.float64)
        iters = np.zeros(N_SAMPLES, dtype=np.int32)
        fp_err = np.zeros(N_SAMPLES, dtype=np.float64)

        for i in range(N_SAMPLES):
            u_pred, n_it, fe = iter_recast_fno(
                model, device, S_norm_all[i], beta, alpha, xcoord, umin, umax, TOL, MAX_ITERS
            )
            u_fno[i] = u_pred
            iters[i] = n_it
            fp_err[i] = fe

        # --- Relative errors ---
        rel_l2 = np.zeros(N_SAMPLES, dtype=np.float64)
        rel_pt = np.zeros(N_SAMPLES, dtype=np.float64)
        for i in range(N_SAMPLES):
            diff = u_fno[i] - u_num[i]
            norm_num = float(np.linalg.norm(u_num[i]))
            rel_l2[i] = float(np.linalg.norm(diff)) / (norm_num + EPS)
            mean_abs_num = float(np.mean(np.abs(u_num[i])))
            rel_pt[i] = float(np.mean(np.abs(diff))) / (mean_abs_num + EPS)

        rel_l2_mean[ib] = np.mean(rel_l2)
        rel_pt_mean[ib] = np.mean(rel_pt)

        print(f"    rel L2 = {rel_l2_mean[ib]:.6e},  rel ptwise = {rel_pt_mean[ib]:.6e},  mean iters = {np.mean(iters):.1f}")

        results[beta_key] = dict(
            u_num=u_num, u_fno=u_fno, rel_l2=rel_l2, rel_pt=rel_pt,
            iters=iters, fp_err=fp_err, beta=beta, alpha=alpha,
        )

    return rel_l2_mean, rel_pt_mean, results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def load_results_from_h5(h5_path: str):
    """Load pre-computed mean error arrays from existing HDF5."""
    with h5py.File(h5_path, "r") as f:
        beta_arr = f["beta_list"][:]
        rel_l2_m1 = f["model1/rel_l2_mean"][:]
        rel_pt_m1 = f["model1/rel_pt_mean"][:]
        rel_l2_m2 = f["model2/rel_l2_mean"][:]
        rel_pt_m2 = f["model2/rel_pt_mean"][:]
    return beta_arr, rel_l2_m1, rel_pt_m1, rel_l2_m2, rel_pt_m2


def plot_comparison(beta_arr, rel_l2_m1, rel_l2_m2, out_fig):

    fig, ax = plt.subplots(figsize=(12, 5))

    # Colors
    c1 = "#2563EB"   # Model1
    c2 = "#059669"   # Model2
    c_mark = "#DC2626"

    pct = 100.0
    rl2_m1 = rel_l2_m1 * pct
    rl2_m2 = rel_l2_m2 * pct

    # ---- Error curves (diamond markers) ----
    ax.plot(
        beta_arr,
        rl2_m1,
        linestyle="-",
        marker="D",
        color=c1,
        linewidth=2.2,
        markersize=6,
        label="Model 1"
    )

    ax.plot(
        beta_arr,
        rl2_m2,
        linestyle="-",
        marker="D",
        color=c2,
        linewidth=2.2,
        markersize=6,
        label="Model 2"
    )

    beta_list_py = list(beta_arr)

    # ---- Canonical marker (Model1) ----
    idx_canon = min(range(len(beta_list_py)),
                    key=lambda i: abs(beta_list_py[i] - 0.785))

    ax.scatter(
        beta_arr[idx_canon],
        rl2_m1[idx_canon],
        marker="*",
        s=350,
        color=c_mark,
        edgecolor="k",
        linewidth=0.8,
        zorder=6,
        label=r"$k^*$"
    )

    # ---- New training point marker (Model2) ----
    idx_new = min(range(len(beta_list_py)),
                  key=lambda i: abs(beta_list_py[i] - 0.05))

    ax.scatter(
        beta_arr[idx_new],
        rl2_m2[idx_new],
        marker="o",
        s=120,
        color=c_mark,
        edgecolor="k",
        linewidth=0.8,
        zorder=6,
        label=r"$k_{new}$"
    )

    # ---- Log scale ----
    ax.set_xscale("log")
    ax.set_yscale("log")

    # ---- Custom k ticks ----
    xticks = [0.01, 0.05, 0.1, 0.5, 0.785, 1, 10]
    xticklabels = ["0.01", "0.05", "0.1", "0.5", "0.785", "1", "10"]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=14)

    # Highlight canonical tick
    for label, val in zip(ax.get_xticklabels(), xticks):
        if abs(val - 0.785) < 1e-12:
            label.set_color("red")
            label.set_fontweight("bold")

    # ---- Axis labels ----
    ax.set_xlabel("k", fontsize=18, fontweight="bold")
    ax.set_ylabel("Relative L2 Error (%)", fontsize=18, fontweight="bold")

    # ---- Title ----
    ax.set_title(
        r"Reaction–Diffusion Recast Test ($-u_{xx}+ku=S$)",
        fontsize=20,
        fontweight="bold"
    )

    # ---- Tick sizes ----
    ax.tick_params(axis="both", labelsize=14)

    # ---- Grid ----
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15, linestyle="--")

    # ---- Legend ----
    ax.legend(fontsize=13, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)

    print(f"[Done] Saved figure: {out_fig}")

def main():
    beta_arr = np.array(k_list, dtype=np.float64)

    # ─── Check for existing results ───
    if os.path.isfile(OUT_H5):
        print(f"[Info] Found existing results: {OUT_H5}. Skipping computation, loading directly.")
        beta_arr, rel_l2_m1, rel_pt_m1, rel_l2_m2, rel_pt_m2 = load_results_from_h5(OUT_H5)
        plot_comparison(beta_arr, rel_l2_m1, rel_l2_m2, OUT_FIG)
        return

    # ─── Compute from scratch ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    # Load u ranges
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
    print("\n===== Model 1 =====")
    rel_l2_m1, rel_pt_m1, res1 = run_one_model(model1, device, u_min1, u_max1, xcoord, S_norm_all, "M1")

    print("\n===== Model 2 =====")
    rel_l2_m2, rel_pt_m2, res2 = run_one_model(model2, device, u_min2, u_max2, xcoord, S_norm_all, "M2")

    # ─── Save HDF5 ───
    with h5py.File(OUT_H5, "w") as f:
        f.attrs["seed"] = SEED
        f.attrs["N_samples"] = N_SAMPLES
        f.attrs["beta_star"] = beta_star
        f.attrs["L"] = L
        f.attrs["dx"] = dx
        f.attrs["N"] = N
        f.attrs["tol"] = TOL
        f.attrs["max_iters"] = MAX_ITERS

        f.create_dataset("beta_list", data=beta_arr)
        f.create_dataset("damping_list", data=np.array(damping_list))
        f.create_dataset("x_phys", data=x)
        f.create_dataset("sources_raw", data=S_raw_all)
        f.create_dataset("sources_norm", data=S_norm_all)
        f.create_dataset("sources_factor", data=S_factor_all)

        for tag, res, rl2, rpt, um, ux in [
            ("model1", res1, rel_l2_m1, rel_pt_m1, u_min1, u_max1),
            ("model2", res2, rel_l2_m2, rel_pt_m2, u_min2, u_max2),
        ]:
            gm = f.create_group(tag)
            gm.attrs["u_min"] = um
            gm.attrs["u_max"] = ux
            gm.create_dataset("rel_l2_mean", data=rl2)
            gm.create_dataset("rel_pt_mean", data=rpt)

            for beta_key, rd in res.items():
                gb = gm.create_group(beta_key)
                gb.attrs["beta"] = rd["beta"]
                gb.attrs["alpha"] = rd["alpha"]
                gb.create_dataset("numerical", data=rd["u_num"])
                gb.create_dataset("FNO", data=rd["u_fno"])
                gb.create_dataset("rel_l2", data=rd["rel_l2"])
                gb.create_dataset("rel_pt", data=rd["rel_pt"])
                gb.create_dataset("iters", data=rd["iters"])
                gb.create_dataset("fp_err", data=rd["fp_err"])

    print(f"\n[Done] Saved HDF5: {OUT_H5}")

    # ─── Plot ───
    plot_comparison(beta_arr, rel_l2_m1, rel_l2_m2, OUT_FIG)


if __name__ == "__main__":
    main()