#!/usr/bin/env python3
"""
steady_vorticity_ns_dataset.py

Generate dataset for 2D steady vorticity Navier–Stokes on the unit torus (0,1)^2:

    u · ∇ω = (1/Re) Δω + S
    Δψ = -ω,  u = ∇^⊥ψ = (∂y ψ, -∂x ψ)

Numerics:
- Pseudo-spectral FFT on 128x128 grid, doubly periodic
- Nonlinearity in physical space (pseudo-spectral)
- 2/3 de-aliasing
- Steady state obtained by pseudo-time marching with IMEX:
    (I - dt/Re Δ) ω^{n+1} = ω^n - dt (u·∇ω)^n + dt S

Outputs a single HDF5 with datasets:
- S     : (N_samples, 128, 128)
- omega : (N_samples, 128, 128)
- u     : (N_samples, 128, 128, 2)  where u[...,0]=u_x, u[...,1]=u_y
Also saves global min/max as file attributes + per-sample min/max arrays.

Additionally, saves three diagnostic figures for samples #1, #20, #100 (1-indexed):
Each figure has two panels: source S and solution omega.

Dependencies: numpy, h5py, matplotlib
"""

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt

# -----------------------------
# User settings
# -----------------------------
OUT_H5 = "data_canonical.h5"

N_SAMPLES = 200
N = 128
RE = 250.0
SEED = 1234

# Random source (smooth, band-limited)
K_MIN = 2
K_MAX = 21        # ~21 for N=128 (smooth, no sharp vortices)
P_SPEC = 2.0              # power spectrum ~ |k|^{-P_SPEC}
SOURCE_STD_TARGET = 1.0   # normalize each source to std=1

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
PLOT_SAMPLE_IDS_1INDEXED = [1, 20, 100]  # requested (1-indexed)
PLOT_DPI = 200


# -----------------------------
# Spectral grids / operators
# -----------------------------
def make_spectral_operators(n: int):
    # FFT frequencies on (0,1): k in {...,-2,-1,0,1,2,...}
    k1 = np.fft.fftfreq(n, d=1.0 / n)  # integers as floats
    kx, ky = np.meshgrid(k1, k1, indexing="ij")

    two_pi = 2.0 * np.pi
    ikx = 1j * two_pi * kx
    iky = 1j * two_pi * ky

    k2 = kx**2 + ky**2
    ksq_phys = (two_pi**2) * k2
    lap_symbol = -ksq_phys  # Δ in Fourier

    inv_ksq = np.zeros_like(ksq_phys)
    inv_ksq[ksq_phys != 0] = 1.0 / ksq_phys[ksq_phys != 0]

    # 2/3 de-aliasing mask: keep |k_i| <= n/3
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
    """
    Fast, stable band-limited random source:
    - sample real white noise in physical space
    - FFT -> apply band-limited power-law spectral filter
    - enforce zero mean
    - IFFT -> real source
    - normalize to std=SOURCE_STD_TARGET
    """
    noise = rng.standard_normal((n, n)).astype(np.float64)
    noise_hat = fft2(noise)

    k_mag = np.sqrt(kx**2 + ky**2)
    band = (k_mag >= K_MIN) & (k_mag <= K_MAX)

    filt = np.zeros_like(k_mag, dtype=np.float64)
    # Multiply amplitude by |k|^{-P_SPEC/2} => power ~ |k|^{-P_SPEC}
    filt[band] = (k_mag[band] + 1e-12) ** (-0.5 * P_SPEC)

    # keep within dealias region too
    filt *= dealias.astype(np.float64)

    S_hat = noise_hat * filt
    S_hat[0, 0] = 0.0  # zero mean

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
def solve_steady_vorticity(S, ikx, iky, lap_symbol, inv_ksq, dealias):
    """
    Pseudo-time marching IMEX:
        (I - dt/Re Δ) ω^{n+1} = ω^n - dt N(ω^n) + dt S
    with N(ω) = u·∇ω, u from Δψ=-ω, u=∇^⊥ψ.

    Returns omega, ux, uy, nsteps, rel_resid
    """
    n = S.shape[0]
    omega = np.zeros((n, n), dtype=np.float64)

    S_hat = fft2(S)
    S_hat[~dealias] = 0.0

    denom = 1.0 - (DT / RE) * lap_symbol
    denom[0, 0] = 1.0

    S_norm = np.linalg.norm(S.ravel()) + EPS

    for step in range(1, MAX_STEPS + 1):
        omega_hat = fft2(omega)

        # Streamfunction
        psi_hat = omega_hat * inv_ksq
        psi_hat[0, 0] = 0.0

        # Velocity
        ux = np.real(ifft2(iky * psi_hat))
        uy = np.real(ifft2(-ikx * psi_hat))

        # Grad omega
        wx = np.real(ifft2(ikx * omega_hat))
        wy = np.real(ifft2(iky * omega_hat))

        # Nonlinear term
        Nl = ux * wx + uy * wy
        Nl_hat = fft2(Nl)
        Nl_hat[~dealias] = 0.0

        # IMEX update
        rhs_hat = omega_hat - DT * Nl_hat + DT * S_hat
        omega_hat_new = rhs_hat / denom
        omega_hat_new[0, 0] = 0.0

        omega = np.real(ifft2(omega_hat_new))
        omega -= omega.mean()  # enforce zero mean

        if step % CHECK_EVERY == 0:
            # residual r = u·∇ω - (1/Re)Δω - S
            omega_hat = omega_hat_new
            psi_hat = omega_hat * inv_ksq
            psi_hat[0, 0] = 0.0
            ux = np.real(ifft2(iky * psi_hat))
            uy = np.real(ifft2(-ikx * psi_hat))
            wx = np.real(ifft2(ikx * omega_hat))
            wy = np.real(ifft2(iky * omega_hat))
            Nl = ux * wx + uy * wy
            lap_omega = np.real(ifft2(lap_symbol * omega_hat))
            resid = Nl - (1.0 / RE) * lap_omega - S
            rel = np.linalg.norm(resid.ravel()) / S_norm
            if rel < TOL_REL_RESID:
                return omega, ux, uy, step, rel

    # not converged: return last iterate
    omega_hat = fft2(omega)
    psi_hat = omega_hat * inv_ksq
    psi_hat[0, 0] = 0.0
    ux = np.real(ifft2(iky * psi_hat))
    uy = np.real(ifft2(-ikx * psi_hat))
    wx = np.real(ifft2(ikx * omega_hat))
    wy = np.real(ifft2(iky * omega_hat))
    Nl = ux * wx + uy * wy
    lap_omega = np.real(ifft2(lap_symbol * omega_hat))
    resid = Nl - (1.0 / RE) * lap_omega - S
    rel = np.linalg.norm(resid.ravel()) / (np.linalg.norm(S.ravel()) + EPS)

    return omega, ux, uy, MAX_STEPS, rel


def save_sample_figure(sample_id_1idx, S, omega, out_prefix):
    """Save a two-panel figure: left=source, right=solution."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = axes[0].imshow(S, origin="lower")
    axes[0].set_title(f"Source S (sample {sample_id_1idx})")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(omega, origin="lower")
    axes[1].set_title(f"Solution ω (steady, Re={RE:g})")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    out_png = f"{out_prefix}_sample{sample_id_1idx:03d}.png"
    fig.savefig(out_png, dpi=PLOT_DPI)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    kx, ky, ikx, iky, lap_symbol, inv_ksq, dealias = make_spectral_operators(N)

    # Which samples to plot (convert to 0-index)
    plot_idx_set = {sid - 1 for sid in PLOT_SAMPLE_IDS_1INDEXED if 1 <= sid <= N_SAMPLES}
    plot_cache = {}  # idx -> (S, omega) in float64 for plotting

    # Prepare HDF5
    with h5py.File(OUT_H5, "w") as f:
        f.attrs["Re"] = float(RE)
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
        du = f.create_dataset(
            "u", shape=(N_SAMPLES, N, N, 2), dtype=DTYPE_STORE,
            compression=COMPRESSION, compression_opts=COMP_LEVEL
        )

        # Per-sample min/max
        s_min = f.create_dataset("S_min", shape=(N_SAMPLES,), dtype=np.float32)
        s_max = f.create_dataset("S_max", shape=(N_SAMPLES,), dtype=np.float32)
        w_min = f.create_dataset("omega_min", shape=(N_SAMPLES,), dtype=np.float32)
        w_max = f.create_dataset("omega_max", shape=(N_SAMPLES,), dtype=np.float32)
        u_min = f.create_dataset("u_min", shape=(N_SAMPLES,), dtype=np.float32)
        u_max = f.create_dataset("u_max", shape=(N_SAMPLES,), dtype=np.float32)
        iters = f.create_dataset("solver_steps", shape=(N_SAMPLES,), dtype=np.int32)
        relres = f.create_dataset("solver_relres", shape=(N_SAMPLES,), dtype=np.float32)

        # Global min/max trackers
        gSmin, gSmax = np.inf, -np.inf
        gWmin, gWmax = np.inf, -np.inf
        gUmin, gUmax = np.inf, -np.inf

        t0 = time.time()
        for i in range(N_SAMPLES):
            S = generate_source(N, kx, ky, dealias, rng)
            omega, ux, uy, nsteps, rrel = solve_steady_vorticity(
                S, ikx, iky, lap_symbol, inv_ksq, dealias
            )
            U = np.stack([ux, uy], axis=-1)

            # Store arrays
            dS[i] = S.astype(DTYPE_STORE)
            domega[i] = omega.astype(DTYPE_STORE)
            du[i] = U.astype(DTYPE_STORE)

            # min/max
            smin_i, smax_i = float(S.min()), float(S.max())
            wmin_i, wmax_i = float(omega.min()), float(omega.max())
            umin_i, umax_i = float(U.min()), float(U.max())

            s_min[i], s_max[i] = smin_i, smax_i
            w_min[i], w_max[i] = wmin_i, wmax_i
            u_min[i], u_max[i] = umin_i, umax_i
            iters[i] = int(nsteps)
            relres[i] = float(rrel)

            gSmin, gSmax = min(gSmin, smin_i), max(gSmax, smax_i)
            gWmin, gWmax = min(gWmin, wmin_i), max(gWmax, wmax_i)
            gUmin, gUmax = min(gUmin, umin_i), max(gUmax, umax_i)

            # cache plots if requested
            if i in plot_idx_set:
                plot_cache[i] = (S.copy(), omega.copy())

            if (i + 1) % 25 == 0 or i == 0:
                elapsed = time.time() - t0
                print(f"[{i+1:4d}/{N_SAMPLES}] steps={nsteps:4d} relres={rrel:.2e}  elapsed={elapsed:.1f}s")

        # Save global min/max
        f.attrs["S_global_min"] = float(gSmin)
        f.attrs["S_global_max"] = float(gSmax)
        f.attrs["omega_global_min"] = float(gWmin)
        f.attrs["omega_global_max"] = float(gWmax)
        f.attrs["u_global_min"] = float(gUmin)
        f.attrs["u_global_max"] = float(gUmax)

    print(f"[DONE] Wrote dataset: {OUT_H5}")

    # Save figures after file is written
    out_prefix = "steady_ns_Re50"
    for sid in PLOT_SAMPLE_IDS_1INDEXED:
        idx = sid - 1
        if idx in plot_cache:
            S, omega = plot_cache[idx]
            save_sample_figure(sid, S, omega, out_prefix)
            print(f"[DONE] Saved figure: {out_prefix}_sample{sid:03d}.png")
        else:
            print(f"[WARN] Sample {sid} not available (out of range or not cached).")


if __name__ == "__main__":
    main()
