#!/usr/bin/env python3
"""
VorticityNS_2D_parametric.py

Generate dataset for 2D steady vorticity Navier-Stokes on the unit torus (0,1)^2
with VARIABLE Reynolds number:

    u · nabla omega = (1/Re) Delta omega + S
    Delta psi = -omega,  u = nabla^perp psi

Each sample draws Re ~ Uniform[RE_MIN, RE_MAX].

Numerics:
- Pseudo-spectral FFT on 128x128 grid, doubly periodic
- 2/3 de-aliasing
- IMEX pseudo-time marching to steady state

Outputs HDF5 with datasets:
- S      : (N_samples, 128, 128)     source field
- omega  : (N_samples, 128, 128)     vorticity solution
- Re     : (N_samples, 128, 128)     Reynolds number as 2D constant field
- u      : (N_samples, 128, 128, 2)  velocity (u_x, u_y)

Global min/max saved as file attributes (including Re_global_min/max = RE_MIN/RE_MAX).
"""

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt

# -----------------------------
# User settings
# -----------------------------
OUT_H5 = "data_parametric.h5"

N_SAMPLES = 200
N = 128
RE_MIN = 200.0
RE_MAX = 300.0
SEED = 1234

# Random source (smooth, band-limited)
K_MIN = 2
K_MAX = 21        # ~21 for N=128
P_SPEC = 2.0              # power spectrum ~ |k|^{-P_SPEC}
SOURCE_STD_TARGET = 1.0

# Pseudo-time solver
DT = 0.2
MAX_STEPS = 2500
CHECK_EVERY = 20
TOL_REL_RESID = 1e-6
EPS = 1e-12

# HDF5 compression
COMPRESSION = "gzip"
COMP_LEVEL = 4

# Storage dtype
DTYPE_STORE = np.float32

# Plot settings
PLOT_SAMPLE_IDS_1INDEXED = [1, 20, 100]
PLOT_DPI = 200


# -----------------------------
# Spectral grids / operators
# -----------------------------
def make_spectral_operators(n: int):
    k1 = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(k1, k1, indexing="ij")

    two_pi = 2.0 * np.pi
    ikx = 1j * two_pi * kx
    iky = 1j * two_pi * ky

    ksq_phys = (two_pi ** 2) * (kx ** 2 + ky ** 2)
    lap_symbol = -ksq_phys

    inv_ksq = np.zeros_like(ksq_phys)
    inv_ksq[ksq_phys != 0] = 1.0 / ksq_phys[ksq_phys != 0]

    kx_abs = np.abs(kx)
    ky_abs = np.abs(ky)
    k_cut = n / 3.0
    dealias = (kx_abs <= k_cut) & (ky_abs <= k_cut)

    return kx, ky, ikx, iky, lap_symbol, inv_ksq, dealias


def fft2(a):
    return np.fft.fft2(a)


def ifft2(a_hat):
    return np.fft.ifft2(a_hat)


# -----------------------------
# Random source generation
# -----------------------------
def generate_source(n, kx, ky, dealias, rng):
    noise = rng.standard_normal((n, n)).astype(np.float64)
    noise_hat = fft2(noise)

    k_mag = np.sqrt(kx ** 2 + ky ** 2)
    band = (k_mag >= K_MIN) & (k_mag <= K_MAX)

    filt = np.zeros_like(k_mag, dtype=np.float64)
    filt[band] = (k_mag[band] + 1e-12) ** (-0.5 * P_SPEC)
    filt *= dealias.astype(np.float64)

    S_hat = noise_hat * filt
    S_hat[0, 0] = 0.0

    S = np.real(ifft2(S_hat))
    S -= S.mean()

    s_std = S.std()
    if s_std < 1e-14:
        return generate_source(n, kx, ky, dealias, rng)

    S *= (SOURCE_STD_TARGET / s_std)
    return S


# -----------------------------
# NS steady solver (pseudo-time marching)
# -----------------------------
def solve_steady_vorticity(S, Re, ikx, iky, lap_symbol, inv_ksq, dealias):
    """
    Same IMEX solver but with per-sample Re.
    """
    n = S.shape[0]
    omega = np.zeros((n, n), dtype=np.float64)

    S_hat = fft2(S)
    S_hat[~dealias] = 0.0

    denom = 1.0 - (DT / Re) * lap_symbol
    denom[0, 0] = 1.0

    S_norm = np.linalg.norm(S.ravel()) + EPS

    for step in range(1, MAX_STEPS + 1):
        omega_hat = fft2(omega)

        psi_hat = omega_hat * inv_ksq
        psi_hat[0, 0] = 0.0

        ux = np.real(ifft2(iky * psi_hat))
        uy = np.real(ifft2(-ikx * psi_hat))

        wx = np.real(ifft2(ikx * omega_hat))
        wy = np.real(ifft2(iky * omega_hat))

        Nl = ux * wx + uy * wy
        Nl_hat = fft2(Nl)
        Nl_hat[~dealias] = 0.0

        rhs_hat = omega_hat - DT * Nl_hat + DT * S_hat
        omega_hat_new = rhs_hat / denom
        omega_hat_new[0, 0] = 0.0

        omega = np.real(ifft2(omega_hat_new))
        omega -= omega.mean()

        if step % CHECK_EVERY == 0:
            omega_hat = omega_hat_new
            psi_hat = omega_hat * inv_ksq
            psi_hat[0, 0] = 0.0
            ux = np.real(ifft2(iky * psi_hat))
            uy = np.real(ifft2(-ikx * psi_hat))
            wx = np.real(ifft2(ikx * omega_hat))
            wy = np.real(ifft2(iky * omega_hat))
            Nl = ux * wx + uy * wy
            lap_omega = np.real(ifft2(lap_symbol * omega_hat))
            resid = Nl - (1.0 / Re) * lap_omega - S
            rel = np.linalg.norm(resid.ravel()) / S_norm
            if rel < TOL_REL_RESID:
                return omega, ux, uy, step, rel

    # not converged
    omega_hat = fft2(omega)
    psi_hat = omega_hat * inv_ksq
    psi_hat[0, 0] = 0.0
    ux = np.real(ifft2(iky * psi_hat))
    uy = np.real(ifft2(-ikx * psi_hat))
    wx = np.real(ifft2(ikx * omega_hat))
    wy = np.real(ifft2(iky * omega_hat))
    Nl = ux * wx + uy * wy
    lap_omega = np.real(ifft2(lap_symbol * omega_hat))
    resid = Nl - (1.0 / Re) * lap_omega - S
    rel = np.linalg.norm(resid.ravel()) / (np.linalg.norm(S.ravel()) + EPS)

    return omega, ux, uy, MAX_STEPS, rel


def save_sample_figure(sample_id_1idx, S, omega, Re_val, out_prefix):
    """Save a two-panel figure: left=source, right=solution."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = axes[0].imshow(S, origin="lower")
    axes[0].set_title(f"Source S (sample {sample_id_1idx})")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(omega, origin="lower")
    axes[1].set_title(f"Solution omega (Re={Re_val:.1f})")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    out_png = f"{out_prefix}_sample{sample_id_1idx:03d}.png"
    fig.savefig(out_png, dpi=PLOT_DPI)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    kx, ky, ikx, iky, lap_symbol, inv_ksq, dealias = make_spectral_operators(N)

    plot_idx_set = {sid - 1 for sid in PLOT_SAMPLE_IDS_1INDEXED if 1 <= sid <= N_SAMPLES}
    plot_cache = {}

    with h5py.File(OUT_H5, "w") as f:
        f.attrs["Re_min"] = float(RE_MIN)
        f.attrs["Re_max"] = float(RE_MAX)
        f.attrs["grid_n"] = int(N)
        f.attrs["domain"] = "(0,1)^2 periodic"
        f.attrs["K_MIN"] = int(K_MIN)
        f.attrs["K_MAX"] = int(K_MAX)
        f.attrs["P_SPEC"] = float(P_SPEC)
        f.attrs["SOURCE_STD_TARGET"] = float(SOURCE_STD_TARGET)
        f.attrs["DT"] = float(DT)
        f.attrs["MAX_STEPS"] = int(MAX_STEPS)
        f.attrs["TOL_REL_RESID"] = float(TOL_REL_RESID)

        dS = f.create_dataset(
            "S", shape=(N_SAMPLES, N, N), dtype=DTYPE_STORE,
            compression=COMPRESSION, compression_opts=COMP_LEVEL
        )
        domega = f.create_dataset(
            "omega", shape=(N_SAMPLES, N, N), dtype=DTYPE_STORE,
            compression=COMPRESSION, compression_opts=COMP_LEVEL
        )
        # Re stored as 2D constant field for direct model ingestion
        dRe = f.create_dataset(
            "Re", shape=(N_SAMPLES, N, N), dtype=DTYPE_STORE,
            compression=COMPRESSION, compression_opts=COMP_LEVEL
        )
        du = f.create_dataset(
            "u", shape=(N_SAMPLES, N, N, 2), dtype=DTYPE_STORE,
            compression=COMPRESSION, compression_opts=COMP_LEVEL
        )

        # Per-sample scalar Re
        dRe_scalar = f.create_dataset("Re_scalar", shape=(N_SAMPLES,), dtype=np.float32)

        # Per-sample min/max
        s_min_ds = f.create_dataset("S_min", shape=(N_SAMPLES,), dtype=np.float32)
        s_max_ds = f.create_dataset("S_max", shape=(N_SAMPLES,), dtype=np.float32)
        w_min_ds = f.create_dataset("omega_min", shape=(N_SAMPLES,), dtype=np.float32)
        w_max_ds = f.create_dataset("omega_max", shape=(N_SAMPLES,), dtype=np.float32)
        u_min_ds = f.create_dataset("u_min", shape=(N_SAMPLES,), dtype=np.float32)
        u_max_ds = f.create_dataset("u_max", shape=(N_SAMPLES,), dtype=np.float32)
        iters = f.create_dataset("solver_steps", shape=(N_SAMPLES,), dtype=np.int32)
        relres = f.create_dataset("solver_relres", shape=(N_SAMPLES,), dtype=np.float32)

        # Global min/max trackers
        gSmin, gSmax = np.inf, -np.inf
        gWmin, gWmax = np.inf, -np.inf
        gUmin, gUmax = np.inf, -np.inf

        t0 = time.time()
        for i in range(N_SAMPLES):
            # Sample Re for this instance
            Re_i = rng.uniform(RE_MIN, RE_MAX)

            S = generate_source(N, kx, ky, dealias, rng)
            omega, ux, uy, nsteps, rrel = solve_steady_vorticity(
                S, Re_i, ikx, iky, lap_symbol, inv_ksq, dealias
            )
            U = np.stack([ux, uy], axis=-1)

            dS[i] = S.astype(DTYPE_STORE)
            domega[i] = omega.astype(DTYPE_STORE)
            du[i] = U.astype(DTYPE_STORE)

            # Store Re as 2D constant field
            dRe[i] = np.full((N, N), Re_i, dtype=DTYPE_STORE)
            dRe_scalar[i] = float(Re_i)

            smin_i, smax_i = float(S.min()), float(S.max())
            wmin_i, wmax_i = float(omega.min()), float(omega.max())
            umin_i, umax_i = float(U.min()), float(U.max())

            s_min_ds[i], s_max_ds[i] = smin_i, smax_i
            w_min_ds[i], w_max_ds[i] = wmin_i, wmax_i
            u_min_ds[i], u_max_ds[i] = umin_i, umax_i
            iters[i] = int(nsteps)
            relres[i] = float(rrel)

            gSmin, gSmax = min(gSmin, smin_i), max(gSmax, smax_i)
            gWmin, gWmax = min(gWmin, wmin_i), max(gWmax, wmax_i)
            gUmin, gUmax = min(gUmin, umin_i), max(gUmax, umax_i)

            if i in plot_idx_set:
                plot_cache[i] = (S.copy(), omega.copy(), Re_i)

            if (i + 1) % 25 == 0 or i == 0:
                elapsed = time.time() - t0
                print(f"[{i+1:4d}/{N_SAMPLES}] Re={Re_i:.1f} steps={nsteps:4d} "
                      f"relres={rrel:.2e}  elapsed={elapsed:.1f}s")

        # Global min/max attributes
        f.attrs["S_global_min"] = float(gSmin)
        f.attrs["S_global_max"] = float(gSmax)
        f.attrs["omega_global_min"] = float(gWmin)
        f.attrs["omega_global_max"] = float(gWmax)
        f.attrs["u_global_min"] = float(gUmin)
        f.attrs["u_global_max"] = float(gUmax)
        # Re global min/max are known exactly
        f.attrs["Re_global_min"] = float(RE_MIN)
        f.attrs["Re_global_max"] = float(RE_MAX)

    print(f"[DONE] Wrote dataset: {OUT_H5}")

    out_prefix = "steady_ns_ReVar"
    for sid in PLOT_SAMPLE_IDS_1INDEXED:
        idx = sid - 1
        if idx in plot_cache:
            S, omega, Re_val = plot_cache[idx]
            save_sample_figure(sid, S, omega, Re_val, out_prefix)
            print(f"[DONE] Saved figure: {out_prefix}_sample{sid:03d}.png")
        else:
            print(f"[WARN] Sample {sid} not available.")


if __name__ == "__main__":
    main()
