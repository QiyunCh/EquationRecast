# ReactionDiffusion.py
#
# Dataset generator for 1D Reaction–Diffusion (shifted Poisson) with Dirichlet BC:
#
#     -u''(x) + beta * u(x) = S(x),   x in (0, L)
#     u(0) = u(L) = 0
#
# This file is designed to be DROP-IN compatible with your existing FNO1D.py + Train.py:
# - Saves an HDF5 file with datasets:
#       "source"   : (n_samples, 201)   (each source rescaled to [-1, 1] exactly like Helmholtz.py)
#       "solution" : (n_samples, 201)
# - Saves attributes:
#       L, dx, N, beta, solution_min, solution_max
#
# note:
# - Sources are sampled as GRF using the same squared-exponential covariance + Cholesky sampling
#   as Helmholtz.py.
# - Source rescaling uses the SAME function structure as Helmholtz.py: rescale_to_minus1_1(S).
#
# Usage:
#   python ReactionDiffusion.py
#
# Output:
#   data_beta_1.000.h5   (by default)
#   sources_solutions_samples.png

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# ---------------------------
# Gaussian Random Field Source Sampling (same as Helmholtz.py)
# ---------------------------
def se_cholesky(x, l=0.2, sigma=1.0, jitter=1e-10):
    """Compute Cholesky factor of squared-exponential covariance matrix."""
    r = np.abs(x[:, None] - x[None, :])
    C = (sigma**2) * np.exp(-(r**2) / (2.0 * l**2))
    C[np.diag_indices_from(C)] += jitter
    return np.linalg.cholesky(C)


def sample_grf_from_cov(Lc, rng):
    """Sample a Gaussian random field using precomputed Cholesky factor."""
    z = rng.normal(0.0, 1.0, size=Lc.shape[0])
    return Lc @ z


def rescale_to_minus1_1(S):
    """Rescale an array to the interval [-1, 1]. (same logic as Helmholtz.py)"""
    Smin, Smax = S.min(), S.max()
    if np.isclose(Smax, Smin):
        return np.zeros_like(S)
    return 2.0 * (S - Smin) / (Smax - Smin) - 1.0


# ---------------------------
# Solver: FD with Dirichlet BCs:  -u'' + beta u = S
# ---------------------------
def solve_fd_dirichlet_scipy_sparse(x, S, beta):
    """
    Solve -u''(x) + beta*u(x) = S(x) with u(0)=u(L)=0 using sparse FD.

    Discretization on interior points (i=1..N-2):
      -u''(x_i) ~ (-u_{i-1} + 2u_i - u_{i+1}) / dx^2

    => A u = S with:
      main = 2/dx^2 + beta
      off  = -1/dx^2
    """
    N = len(x)
    dx = x[1] - x[0]
    M = N - 2
    if M <= 0:
        return np.zeros_like(S)

    main = (2.0 / dx**2 + beta) * np.ones(M)
    off = (-1.0 / dx**2) * np.ones(M - 1)
    A = diags([off, main, off], offsets=[-1, 0, 1], format="csr")

    rhs = S[1:-1]
    u_int = spsolve(A, rhs)

    u = np.zeros_like(S)
    u[1:-1] = u_int
    return u


# ---------------------------
# Dataset generation
# ---------------------------
def generate_dataset(
    beta=1.0,
    n_samples=1000,
    seed=13,
    L=10.0,
    dx=0.05,
    grf_l=0.5,
    grf_sigma=1.0,
    out_path=None,
    make_plot=True,
):
    # grid (match Helmholtz.py defaults)
    N = int(L / dx) + 1
    x = np.linspace(0.0, L, N, endpoint=True)

    rng = np.random.default_rng(seed)
    Lc = se_cholesky(x, l=grf_l, sigma=grf_sigma)

    sources = np.zeros((n_samples, N), dtype=np.float64)
    solutions = np.zeros((n_samples, N), dtype=np.float64)

    print(f"Mesh: L={L}, dx={dx}, N={N} | beta={beta:.6f}")
    print(f"Sampling GRF: n_samples={n_samples}, seed={seed}, l={grf_l}, sigma={grf_sigma}")

    for i in range(n_samples):
        S_raw = sample_grf_from_cov(Lc, rng)
        S = rescale_to_minus1_1(S_raw)  # IMPORTANT: identical style as Helmholtz.py
        u = solve_fd_dirichlet_scipy_sparse(x, S, beta)

        sources[i] = S
        solutions[i] = u

        if (i + 1) % 100 == 0:
            print(f"  generated {i+1}/{n_samples}")

    sol_min = float(np.min(solutions))
    sol_max = float(np.max(solutions))
    print(f"\nGlobal solution range: min={sol_min:+.6f}, max={sol_max:+.6f}")

    if out_path is None:
        out_path = f"data_beta_{beta:.3f}.h5"

    # Save to HDF5 (Train.py compatible)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("source", data=sources)
        f.create_dataset("solution", data=solutions)
        f.attrs["L"] = float(L)
        f.attrs["dx"] = float(dx)
        f.attrs["N"] = int(N)
        f.attrs["beta"] = float(beta)
        f.attrs["solution_min"] = sol_min
        f.attrs["solution_max"] = sol_max
        f.attrs["note"] = (
            "FD Dirichlet solve for 1D reaction-diffusion: -u'' + beta*u = S. "
            f"Sources are GRF (SE kernel) with l={grf_l}, sigma={grf_sigma}. "
            "Each source is rescaled to [-1, 1] using min-max rescaling (as in Helmholtz.py). "
            "solution_min and solution_max give global range across all samples."
        )

    print("\nSaved:")
    print(f"  - HDF5: {out_path}  (datasets: 'source' [n_samples×{N}], 'solution' [n_samples×{N}])")
    print("  - Attributes: L, dx, N, beta, solution_min, solution_max")

    # Optional plot (same spirit as Helmholtz.py)
    if make_plot:
        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        for i in range(3):
            ax1 = axes[i]
            ax2 = ax1.twinx()

            p1, = ax2.plot(x, sources[i], lw=1.2, ls="--", label="source [-1,1]")
            p2, = ax1.plot(x, solutions[i], lw=1.5, label="solution")

            ax1.set_ylabel("solution")
            ax2.set_ylabel("source")
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"Sample #{i+1}")
            ax1.legend([p1, p2], ["source", "solution"], loc="upper right", frameon=False)

        axes[-1].set_xlabel("x")
        plt.tight_layout()
        plt.savefig("sources_solutions_samples.png", dpi=300, bbox_inches="tight")
        plt.show()

    return out_path


if __name__ == "__main__":
    # Default settings chosen to mirror Helmholtz.py as closely as possible.
    # You can change beta and output filename as needed.
    generate_dataset(
        beta=0.785,
        n_samples=1000,
        seed=13,
        L=10.0,
        dx=0.05,
        grf_l=0.5,
        grf_sigma=1.0,
        out_path=None,
        make_plot=True,
    )
