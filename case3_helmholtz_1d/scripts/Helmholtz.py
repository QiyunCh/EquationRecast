import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# ---------------------------
# Gaussian Random Field Source Sampling
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
    """Rescale an array to the interval [-1, 1]."""
    Smin, Smax = S.min(), S.max()
    return 2.0 * (S - Smin) / (Smax - Smin) - 1.0

# ---------------------------
# Solver: FD with Dirichlet BCs:  u'' + k^2 u = -S
# ---------------------------
def solve_fd_dirichlet_scipy_sparse(x, S, k):
    N = len(x)
    dx = x[1] - x[0]
    M = N - 2
    if M <= 0:
        return np.zeros_like(S)

    main = (2.0 / dx**2 - k**2) * np.ones(M)
    off  = (-1.0 / dx**2) * np.ones(M - 1)
    A = diags([off, main, off], offsets=[-1, 0, 1], format="csr")
    b = S[1:-1]
    u_int = spsolve(A, b)

    u = np.zeros_like(S)
    u[1:-1] = u_int
    return u

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    seed = 618

    # Equation in 1D slab:  u'' + k^2 u = - S
    k  = 1.099
    L  = 10.0
    dx = 0.05
    N  = 201
    x = np.linspace(0.0, L, N, endpoint=True)

    # GRF parameters
    rng = np.random.default_rng(seed)
    n_samples = 1000
    l = 0.5
    sigma = 1.0

    # Precompute covariance factor
    Lc = se_cholesky(x, l=l, sigma=sigma)

    sources   = np.zeros((n_samples, N))
    solutions = np.zeros((n_samples, N))

    print(f"Mesh: L={L}, dx={dx}, N={N} | k={k:.3f}")
    print(f"Sampling Gaussian Random Fields (l={l}) → rescale to [-1,1] → solve Helmholtz...")

    for i in range(n_samples):
        # GRF sampling + rescaling
        S = sample_grf_from_cov(Lc, rng)
        S_rescaled = rescale_to_minus1_1(S)

        # Solve
        t0 = time.perf_counter()
        u = solve_fd_dirichlet_scipy_sparse(x, S_rescaled, k)
        t1 = time.perf_counter()

        sources[i, :]   = S_rescaled
        solutions[i, :] = u

        print(f"  sample {i+1}: range=[{S_rescaled.min():+.3f}, {S_rescaled.max():+.3f}], "
              f"solve={1e3*(t1-t0):.2f} ms")

    # ---------------- Compute global min/max ----------------
    sol_min = np.min(solutions)
    sol_max = np.max(solutions)
    print(f"\nGlobal solution range: min={sol_min:+.6f}, max={sol_max:+.6f}")

    # ---------------- Save to HDF5 ----------------
    with h5py.File("data.h5", "w") as f:
        f.create_dataset("source",   data=sources)
        f.create_dataset("solution", data=solutions)
        f.attrs["L"]   = L
        f.attrs["dx"]  = dx
        f.attrs["N"]   = N
        f.attrs["k"]   = k
        f.attrs["solution_min"] = sol_min
        f.attrs["solution_max"] = sol_max
        f.attrs["note"] = (
            "FD Dirichlet solve for 1D Helmholtz: u'' + k^2 u = -S. "
            f"Sources are Gaussian Random Field (length scale l={l}, sigma={sigma}). "
            "Each source is rescaled to the range [-1, 1]. "
            "solution_min and solution_max give global range across all samples."
        )

    print("\nSaved:")
    print("  - HDF5: data.h5  (datasets: 'source' [n_samples×201], 'solution' [n_samples×201])")
    print("  - Attributes: L, dx, N, k, solution_min, solution_max")

    # ---------------- Plot: 3 samples ----------------
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
